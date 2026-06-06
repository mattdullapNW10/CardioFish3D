import tifffile
from pathlib import Path

DEFAULT_TIF = (
    Path(__file__).parent.parent
    / "data/raw/cmlc2_lifeactXnuclear/48hpf"
    / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
)

def extract_imagej_metadata(path):
    with tifffile.TiffFile(path) as tif:
        print("--- TIFF Tags ---")
        for tag in tif.pages[0].tags.values():
            if tag.name not in ["ColorMap", "StripOffsets", "StripByteCounts"]:
                val = str(tag.value)
                if len(val) > 100:
                    print(f"{tag.name}: {val[:100]}... (len {len(val)})")
                else:
                    print(f"{tag.name}: {val}")
        
        print("\n--- ImageJ Metadata ---")
        if tif.imagej_metadata:
            for k, v in tif.imagej_metadata.items():
                if k == 'Info':
                    print("\nFound 'Info' section, scanning for microscope terms...")
                    # The 'Info' section often contains the original lif metadata
                    lines = str(v).split('\n')
                    for line in lines:
                        lower_line = line.lower()
                        if any(term in lower_line for term in ['microscope', 'objective', 'lens', 'numericalaperture', 'na=', 'magnification', 'detector', 'leica', 'model']):
                            print(line)
                else:
                    val = str(v)
                    print(f"{k}: {val[:100]}...")
        else:
            print("No ImageJ metadata found.")

if __name__ == "__main__":
    extract_imagej_metadata(DEFAULT_TIF)
