import os
import httpx
from dptb_pilot.tools.init import mcp
from typing import List, Dict, Any
from pymatgen.core import Structure

# OPTIMADE endpoint for C2DB at DTU
C2DB_BASE_URL = "https://cmr-optimade.fysik.dtu.dk/v1"

@mcp.tool()
def search_c2db(query: str, limit: int = 5) -> str:
    """
    Search for 2D materials in the C2DB database using OPTIMADE API.
    
    Args:
        query: Search query. Supports logical formulas/names (e.g., "MoS2", "Graphene").
               Will be converted to OPTIMADE filter 'chemical_formula_descriptive CONTAINS "query"'.
        limit: Maximum number of results to return (default: 5).
        
    Returns:
        Formatted list of materials with C2DB IDs and properties.
    """
    try:
        url = f"{C2DB_BASE_URL}/structures"
        
        # Construct OPTIMADE filter
        # We assume the user searches by chemical name/formula
        # "chemical_formula_descriptive" is a standard OPTIMADE field
        filter_str = f'chemical_formula_descriptive CONTAINS "{query}"'
        
        params = {
            "filter": filter_str,
            "page_limit": limit,
            "response_fields": "id,chemical_formula_descriptive,chemical_formula_reduced,lattice_vectors"
        }
        
        with httpx.Client(timeout=15.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("data", [])
            
            if not results:
                return f"No 2D materials found in C2DB for query: {query}"
                
            output = f"Found {len(results)} materials in C2DB for '{query}' (showing top {len(results)}):\n\n"
            
            for i, item in enumerate(results):
                c2db_id = item.get("id")
                # Try to get readable formula
                attrs = item.get("attributes", {})
                formula = attrs.get("chemical_formula_descriptive") or attrs.get("chemical_formula_reduced")
                
                output += f"{i+1}. **ID: {c2db_id}** - {formula}\n"
                
            output += "\nTo download, use `download_c2db_structure` with the ID."
            return output

    except httpx.RequestError as e:
        return f"Network error searching C2DB: {str(e)}"
    except Exception as e:
        return f"Error searching C2DB: {str(e)}"

@mcp.tool()
def download_c2db_structure(c2db_id: str, work_path: str = ".") -> str:
    """
    Download a structure from C2DB by its ID and save as CIF.
    
    Args:
        c2db_id: The C2DB ID (e.g., "MoS2-d1t1").
        work_path: The directory to save the file in.
        
    Returns:
        Success message with path.
    """
    try:
        # OPTIMADE /structures/{id} endpoint
        url = f"{C2DB_BASE_URL}/structures/{c2db_id}"
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            
            if response.status_code == 404:
                return f"Error: C2DB ID {c2db_id} not found."
                
            response.raise_for_status()
            
            data = response.json()
            entry = data.get("data")
            if not entry:
                 return f"Error: Empty response for ID {c2db_id}"
                 
            attrs = entry.get("attributes", {})
            
            # Convert OPTIMADE structure to Pymatgen Structure
            # OPTIMADE fields:
            # - lattice_vectors: 3x3 matrix
            # - cartesian_site_positions: Nx3 matrix
            # - species_at_sites: list of strings (element symbols)
            
            lattice = attrs.get("lattice_vectors")
            coords = attrs.get("cartesian_site_positions")
            species = attrs.get("species_at_sites") # e.g., ["Mo", "S", "S"]
            
            if not lattice or not coords or not species:
                return "Error: Incomplete structure data from OPTIMADE API."
                
            # Create Pymatgen structure
            # Note: species_at_sites in OPTIMADE usually maps to element symbols directly
            # but sometimes referencing 'species' list. C2DB usually creates simple species.
            
            struct = Structure(lattice, species, coords, coords_are_cartesian=True)
            
            # Save
            if not os.path.exists(work_path):
                 os.makedirs(work_path, exist_ok=True)
                 
            filename = f"{c2db_id}.cif"
            save_path = os.path.join(work_path, filename)
            
            struct.to(filename=save_path)
            
            return f"Successfully downloaded C2DB structure {c2db_id} to `{save_path}`."

    except Exception as e:
        return f"Error downloading C2DB structure {c2db_id}: {str(e)}"
