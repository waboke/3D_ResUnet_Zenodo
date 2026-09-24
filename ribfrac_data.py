"""Reusable RibFrac data-access utilities."""
from __future__ import annotations
import struct, zipfile, zlib
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import requests

BASE_DIR=Path('/kaggle/working')
IMAGE_DIR=BASE_DIR/'ribfrac_images'; LABEL_DIR=BASE_DIR/'ribfrac_labels'; METADATA_DIR=BASE_DIR/'ribfrac_metadata'
for d in (IMAGE_DIR,LABEL_DIR,METADATA_DIR): d.mkdir(parents=True,exist_ok=True)
CT_ZIP_URL='https://zenodo.org/records/3893508/files/ribfrac-train-images-1.zip?download=1'
LABEL_ZIP_URL='https://zenodo.org/records/3893508/files/ribfrac-train-labels-1.zip?download=1'
INFO_URL='https://zenodo.org/records/3893508/files/ribfrac-train-info-1.csv?download=1'
INFO_PATH=METADATA_DIR/'ribfrac-train-info-1.csv'; LABEL_ZIP_PATH=BASE_DIR/'ribfrac-train-labels-1.zip'

def http_range_get(url,start,end,timeout=120):
    r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=timeout); r.raise_for_status()
    expected=end-start+1
    if r.status_code!=206 or len(r.content)!=expected:
        raise RuntimeError(f'Range request failed: status={r.status_code}, expected={expected}, received={len(r.content)}')
    return r.content

def download_metadata(force=False):
    if INFO_PATH.exists() and not force: return INFO_PATH
    r=requests.get(INFO_URL,timeout=120); r.raise_for_status(); INFO_PATH.write_bytes(r.content); return INFO_PATH

def load_metadata():
    if not INFO_PATH.exists(): download_metadata()
    df=pd.read_csv(INFO_PATH); req={'public_id','label_id','label_code'}
    missing=req-set(df.columns)
    if missing: raise ValueError(f'Metadata missing columns: {sorted(missing)}')
    return df

def download_label_zip(force=False):
    if LABEL_ZIP_PATH.exists() and not force: return LABEL_ZIP_PATH
    r=requests.get(LABEL_ZIP_URL,timeout=300); r.raise_for_status(); LABEL_ZIP_PATH.write_bytes(r.content); return LABEL_ZIP_PATH

def build_label_index():
    if not LABEL_ZIP_PATH.exists(): download_label_zip()
    out={}
    with zipfile.ZipFile(LABEL_ZIP_PATH) as z:
        for member in z.namelist():
            if member.endswith('-label.nii.gz'):
                out[Path(member).name.replace('-label.nii.gz','')]=member
    return out

def get_label_path(patient_id,label_index=None):
    if label_index is None: label_index=build_label_index()
    if patient_id not in label_index: raise ValueError(f'No label file found for {patient_id}')
    member=label_index[patient_id]; out=LABEL_DIR/member; out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        with zipfile.ZipFile(LABEL_ZIP_PATH) as z: z.extract(member,LABEL_DIR)
    return out

def _find_zip64_eocd(zip_url=CT_ZIP_URL,tail_size=65536):
    r=requests.get(zip_url,headers={'Range':f'bytes=-{tail_size}'},timeout=120); r.raise_for_status(); tail=r.content
    p=tail.rfind(b'PK\x06\x07')
    if p<0: raise RuntimeError('ZIP64 locator not found')
    _,_,eocd_offset,_=struct.unpack('<4sIQI',tail[p:p+20])
    r=requests.get(zip_url,headers={'Range':f'bytes={eocd_offset}-{eocd_offset+55}'},timeout=120); r.raise_for_status(); e=r.content
    if e[:4]!=b'PK\x06\x06': raise RuntimeError('ZIP64 EOCD not found')
    return (r, eocd_offset, struct.unpack_from('<Q',e,32)[0], struct.unpack_from('<Q',e,40)[0], struct.unpack_from('<Q',e,48)[0])

def build_ct_zip_index(zip_url=CT_ZIP_URL):
    _,_,total,size,offset=_find_zip64_eocd(zip_url)
    cd=http_range_get(zip_url,offset,offset+size-1); records=[]; pos=0
    while pos<len(cd):
        if cd[pos:pos+4]!=b'PK\x01\x02': raise RuntimeError(f'Invalid central-directory signature at {pos}')
        vals=struct.unpack('<4s6H3I5H2I',cd[pos:pos+46]);
        _,_,_,_,method,_,_,_,cs,us,nl,el,cl,_,_,_,lho=vals
        fs=pos+46; fe=fs+nl; es=fe; ee=es+el; name=cd[fs:fe].decode('utf-8','replace'); extra=cd[es:ee]
        ep=0
        if cs==0xffffffff or us==0xffffffff or lho==0xffffffff:
            while ep+4<=len(extra):
                fid,flen=struct.unpack_from('<HH',extra,ep); data=extra[ep+4:ep+4+flen]
                if fid==1:
                    dp=0
                    if us==0xffffffff: us=struct.unpack_from('<Q',data,dp)[0]; dp+=8
                    if cs==0xffffffff: cs=struct.unpack_from('<Q',data,dp)[0]; dp+=8
                    if lho==0xffffffff: lho=struct.unpack_from('<Q',data,dp)[0]
                    break
                ep+=4+flen
        records.append({'filename':name,'compressed_size':cs,'uncompressed_size':us,'local_header_offset':lho,'compression_method':method})
        pos=ee+cl
    if len(records)!=total: raise RuntimeError(f'Expected {total} entries, parsed {len(records)}')
    return pd.DataFrame(records)

def get_ct_entry(patient_id,ct_zip_index):
    name=f'Part1/{patient_id}-image.nii.gz'; m=ct_zip_index[ct_zip_index.filename==name]
    if m.empty: raise ValueError(f'CT file not found: {name}')
    return m.iloc[0]

def download_ribfrac_image(patient_id,ct_zip_index=None,output_dir=IMAGE_DIR,chunk_size=1024*1024,zip_url=CT_ZIP_URL):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); out=output_dir/f'{patient_id}-image.nii.gz'
    if out.exists(): return out
    if ct_zip_index is None: ct_zip_index=build_ct_zip_index(zip_url)
    e=get_ct_entry(patient_id,ct_zip_index); lho=int(e.local_header_offset); cs=int(e.compressed_size)
    h=http_range_get(zip_url,lho,lho+29)
    if h[:4]!=b'PK\x03\x04': raise RuntimeError('Invalid local ZIP header')
    _,_,_,method,_,_,_,_,_,nl,el=struct.unpack('<4s5H3I2H',h)
    start=lho+30+nl+el; end=start+cs-1; chunks=[]
    for s in range(start,end+1,chunk_size): chunks.append(http_range_get(zip_url,s,min(s+chunk_size-1,end)))
    data=b''.join(chunks)
    if len(data)!=cs: raise RuntimeError('Compressed data size mismatch')
    if method!=8: raise RuntimeError(f'Unexpected compression method {method}')
    raw=zlib.decompress(data,-15)
    if len(raw)!=int(e.uncompressed_size): raise RuntimeError('Decompressed size mismatch')
    out.write_bytes(raw); return out

def load_patient(patient_id,ct_zip_index=None,label_index=None):
    ct_path=download_ribfrac_image(patient_id,ct_zip_index=ct_zip_index); label_path=get_label_path(patient_id,label_index=label_index)
    ct_img=nib.load(str(ct_path)); label_img=nib.load(str(label_path))
    return {'patient_id':patient_id,'ct_path':ct_path,'label_path':label_path,'ct_img':ct_img,'label_img':label_img,'ct_volume':ct_img.get_fdata(dtype=np.float32),'label_volume':label_img.get_fdata(dtype=np.float32)}

def validate_patient(patient):
    c,l=patient['ct_img'],patient['label_img']; cs,ls=c.shape,l.shape; csp,lsp=c.header.get_zooms()[:3],l.header.get_zooms()[:3]
    sm=cs==ls; sp=np.allclose(csp,lsp,atol=1e-5); af=np.allclose(c.affine,l.affine,atol=1e-4)
    if not (sm and sp and af): raise ValueError(f"Spatial validation failed for {patient['patient_id']}")
    return {'patient_id':patient['patient_id'],'shape_match':sm,'spacing_match':sp,'affine_match':af,'ct_shape':cs,'label_shape':ls,'ct_spacing':csp,'label_spacing':lsp,'ct_min':float(np.min(patient['ct_volume'])),'ct_max':float(np.max(patient['ct_volume'])),'label_unique_values':np.unique(patient['label_volume']),'label_nonzero_voxels':int(np.count_nonzero(patient['label_volume']))}

def get_patient_annotation_metadata(patient_id,metadata=None):
    if metadata is None: metadata=load_metadata()
    rows=metadata[metadata.public_id==patient_id].copy()
    if rows.empty: raise ValueError(f'No metadata found for {patient_id}')
    return rows

def initialize_data_access():
    return load_metadata(),build_ct_zip_index(),build_label_index()
