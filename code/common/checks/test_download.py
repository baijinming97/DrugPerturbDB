"""Large source downloads resume only at a verified byte boundary."""
import hashlib
import io
import pytest
from tahoe_repro.sources import download_one

CONTENT=b'0123456789'

def source():
    return dict(id='example',filename='source.bin',bytes=len(CONTENT),sha256=hashlib.sha256(CONTENT).hexdigest(),url='https://example.test/source.bin')

class Response(io.BytesIO):
    def __init__(self,payload,status=206,content_range='bytes 4-9/10'):
        super().__init__(payload);self.status=status;self.headers={'Content-Range':content_range}

def test_range_resume(tmp_path,monkeypatch):
    (tmp_path/'source.bin.partial').write_bytes(CONTENT[:4])
    def open_source(request,timeout):
        assert request.get_header('Range')=='bytes=4-'
        return Response(CONTENT[4:])
    monkeypatch.setattr('urllib.request.urlopen',open_source)
    result=download_one(source(),tmp_path,reserve_bytes=0)
    assert result['downloaded_bytes']==6
    assert (tmp_path/'source.bin').read_bytes()==CONTENT

def test_ignored_range_preserves_partial(tmp_path,monkeypatch):
    partial=tmp_path/'source.bin.partial';partial.write_bytes(CONTENT[:4])
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:Response(CONTENT,200,''))
    with pytest.raises(RuntimeError,match='resume range'):download_one(source(),tmp_path,reserve_bytes=0)
    assert partial.read_bytes()==CONTENT[:4]

def test_complete_partial_is_verified_without_network(tmp_path,monkeypatch):
    (tmp_path/'source.bin.partial').write_bytes(CONTENT)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:pytest.fail('Unexpected download'))
    assert download_one(source(),tmp_path,reserve_bytes=0)['downloaded_bytes']==0

def test_wrong_complete_partial_is_not_promoted(tmp_path):
    partial=tmp_path/'source.bin.partial';partial.write_bytes(b'xxxxxxxxxx')
    with pytest.raises(RuntimeError,match='wrong checksum'):download_one(source(),tmp_path,reserve_bytes=0)
    assert partial.exists() and not (tmp_path/'source.bin').exists()
