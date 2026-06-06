import sys
import re
from pathlib import Path
import tifffile

def extract_specific_metadata(filename: str):
    """
    Given a file name in the raw data directory, extracts the numerical aperture 
    and wavelength from the microscope metadata of the .tif file.
    """
    path = Path(filename)
    
    # Auto-resolve relative to data/raw if it doesn't exist directly
    if not path.is_absolute() and not path.exists():
        raw_data_dir = Path("data/raw")
        resolved_path = raw_data_dir / path
        if resolved_path.exists():
            path = resolved_path

    if not path.exists():
        print(f"Error: File '{path}' does not exist.")
        return

    output_filename = path.stem + "_extracted_params.txt"
    output_path = Path(output_filename)

    print(f"--- Extracting NA and Wavelength for: {path} ---")
    print(f"--- Saving to: {output_path.absolute()} ---")
    
    try:
        with tifffile.TiffFile(path) as tif:
            metadata_text = ""
            
            # Check for ImageJ Info field
            if tif.is_imagej and tif.imagej_metadata and 'Info' in tif.imagej_metadata:
                metadata_text += tif.imagej_metadata['Info'] + "\n"
            
            # Check for OME-XML
            if tif.is_ome:
                metadata_text += str(tif.ome_metadata) + "\n"
                
            if not metadata_text and tif.pages:
                desc = tif.pages[0].tags.get('ImageDescription')
                if desc:
                    metadata_text += str(desc.value) + "\n"
            
            # Use regex to find lines containing numerical aperture or wavelength
            na_lines = []
            wavelength_lines = []
            
            for line in metadata_text.splitlines():
                line_lower = line.lower()
                # Focus on lines related to the current series or general properties
                if "numericalaperture" in line_lower or "numerical aperture" in line_lower:
                    na_lines.append(line.strip())
                elif "wavelength" in line_lower:
                    wavelength_lines.append(line.strip())
            
            # We want to filter out properties from OTHER series if possible
            # We can infer the series name from the file name, e.g. "Series001"
            series_match = re.search(r'(Series\d+)', path.name)
            current_series = series_match.group(1) if series_match else None
            
            def filter_for_series(lines):
                if not current_series:
                    return lines
                
                # Try to find lines strictly for this series
                series_lines = [l for l in lines if current_series in l]
                if series_lines:
                    return series_lines
                # If none found specific to the series, return all
                return lines

            relevant_na = filter_for_series(na_lines)
            relevant_wavelength = filter_for_series(wavelength_lines)

            with open(output_path, "w", encoding="utf-8") as out:
                out.write(f"--- Extracted Parameters for: {path} ---\n\n")
                
                out.write("=== Numerical Aperture ===\n")
                if relevant_na:
                    for line in relevant_na:
                        out.write(f"{line}\n")
                else:
                    out.write("Not found.\n")
                    
                out.write("\n=== Wavelength ===\n")
                if relevant_wavelength:
                    for line in relevant_wavelength:
                        out.write(f"{line}\n")
                else:
                    out.write("Not found.\n")

        print("Success!")

    except Exception as e:
        print(f"Error reading TIFF metadata: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_metadata.py <path_to_tif_file>")
        sys.exit(1)
        
    extract_specific_metadata(sys.argv[1])
