import tifffile
from pathlib import Path

DEFAULT_TIF = (
    Path(__file__).parent.parent
    / "data/raw/cmlc2_lifeactXnuclear/48hpf"
    / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
)

def print_metadata(path):
    print(f"Reading metadata from {path.name}...\n")
    with tifffile.TiffFile(path) as tif:
        # Check for OME metadata
        if tif.is_ome:
            print("--- OME-XML Metadata Found ---")
            ome_xml = tif.ome_metadata
            # Print first 2000 chars of OME XML just to see what's in there
            print(ome_xml[:2000])
            print("...\n")
        
        # Check standard TIFF tags on the first page
        print("--- TIFF Tags (First Page) ---")
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"{name}: {value}")
            
        # If it's a lif export, maybe ImageDescription has useful stuff
        image_desc = tif.pages[0].tags.get("ImageDescription")
        if image_desc:
            print("\n--- Image Description Full Text ---")
            print(image_desc.value[:2000])

if __name__ == "__main__":
    print_metadata(DEFAULT_TIF)
