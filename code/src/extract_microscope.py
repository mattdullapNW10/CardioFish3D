import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_TIF = (
    Path(__file__).parent.parent
    / "data/raw/cmlc2_lifeactXnuclear/48hpf"
    / "16012025_cmlc2_lifeactxnuclear_48hpf.lif - Series005.tif"
)

def extract_microscope_info(path):
    with tifffile.TiffFile(path) as tif:
        if not tif.is_ome:
            print("No OME metadata found.")
            return
            
        ome_xml = tif.ome_metadata
        try:
            root = ET.fromstring(ome_xml)
            # Remove namespace for easier searching
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]
            
            # Find Instrument section
            instruments = root.findall('.//Instrument')
            if not instruments:
                print("No Instrument data found in OME-XML.")
                
            for instrument in instruments:
                print(f"Instrument ID: {instrument.get('ID')}")
                
                microscope = instrument.find('.//Microscope')
                if microscope is not None:
                    print(f"  Microscope:")
                    print(f"    Manufacturer: {microscope.get('Manufacturer', 'N/A')}")
                    print(f"    Model: {microscope.get('Model', 'N/A')}")
                    print(f"    Serial Number: {microscope.get('SerialNumber', 'N/A')}")
                
                objectives = instrument.findall('.//Objective')
                if objectives:
                    print("  Objectives:")
                    for obj in objectives:
                        print(f"    - Manufacturer: {obj.get('Manufacturer', 'N/A')}")
                        print(f"      Model: {obj.get('Model', 'N/A')}")
                        print(f"      NominalMagnification: {obj.get('NominalMagnification', 'N/A')}")
                        print(f"      LensNA: {obj.get('LensNA', 'N/A')}")
                        print(f"      Immersion: {obj.get('Immersion', 'N/A')}")

                detectors = instrument.findall('.//Detector')
                if detectors:
                    print("  Detectors:")
                    for det in detectors:
                        print(f"    - Manufacturer: {det.get('Manufacturer', 'N/A')}")
                        print(f"      Model: {det.get('Model', 'N/A')}")
                        print(f"      Type: {det.get('Type', 'N/A')}")
        except Exception as e:
            print(f"Error parsing XML: {e}")

if __name__ == "__main__":
    extract_microscope_info(DEFAULT_TIF)
