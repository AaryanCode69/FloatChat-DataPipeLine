"""
Export ChromaDB Data to Text File
=================================

This script connects to ChromaDB, retrieves all data from the float_embeddings collection,
and saves it to a comprehensive text file with proper formatting.

Author: FloatChat Data Pipeline
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from ingest.db_handler import ChromaDBHandler

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)


def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata dictionary into readable text."""
    formatted_lines = []
    
    # Essential information first
    if 'float_id' in metadata:
        formatted_lines.append(f"Float ID: {metadata['float_id']}")
    
    if 'source' in metadata:
        formatted_lines.append(f"Source: {metadata['source']}")
    
    if 'data_type' in metadata:
        formatted_lines.append(f"Data Type: {metadata['data_type']}")
    
    # Location information
    location_fields = ['lat_min', 'lat_max', 'lon_min', 'lon_max']
    location_data = {k: v for k, v in metadata.items() if k in location_fields}
    if location_data:
        formatted_lines.append("Location Range:")
        for key, value in location_data.items():
            formatted_lines.append(f"  {key}: {value}")
    
    # Measurement flags
    measurement_flags = ['has_temperature', 'has_salinity', 'has_pressure']
    measurements = {k: v for k, v in metadata.items() if k in measurement_flags}
    if measurements:
        formatted_lines.append("Available Measurements:")
        for key, value in measurements.items():
            measurement_type = key.replace('has_', '').title()
            formatted_lines.append(f"  {measurement_type}: {'Yes' if value else 'No'}")
    
    # Other metadata
    other_fields = [k for k in metadata.keys() if k not in 
                   ['float_id', 'source', 'data_type'] + location_fields + measurement_flags]
    
    if other_fields:
        formatted_lines.append("Additional Information:")
        for key in other_fields:
            value = metadata[key]
            if isinstance(value, (dict, list)):
                formatted_lines.append(f"  {key}: {json.dumps(value, indent=4)}")
            else:
                formatted_lines.append(f"  {key}: {value}")
    
    return '\n'.join(formatted_lines)


def export_chromadb_to_text(output_filename: str = None) -> str:
    """
    Export all ChromaDB data to a text file.
    
    Args:
        output_filename: Name of the output file (optional)
        
    Returns:
        Path to the created file
    """
    # Generate filename if not provided
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"chromadb_export_{timestamp}.txt"
    
    output_path = Path(__file__).parent / output_filename
    
    try:
        print("🔍 Connecting to ChromaDB...")
        
        # Connect to ChromaDB
        chroma = ChromaDBHandler()
        
        # Get the float_embeddings collection
        collection = chroma.client.get_collection("float_embeddings")
        print(f"✓ Connected to collection: float_embeddings")
        
        # Get all documents with metadata
        print("📥 Retrieving all data from ChromaDB...")
        results = collection.get(include=["documents", "metadatas"])
        
        # Get IDs separately if needed
        all_results = collection.get()
        document_ids = all_results['ids'] if 'ids' in all_results else []
        
        total_documents = len(results['documents'])
        print(f"✓ Found {total_documents} documents")
        
        if total_documents == 0:
            print("⚠️  No data found in ChromaDB collection")
            return None
        
        # Write to text file
        print(f"📝 Writing data to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 100 + "\n")
            f.write("CHROMADB DATA EXPORT - FLOAT EMBEDDINGS COLLECTION\n")
            f.write("=" * 100 + "\n")
            f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Collection: float_embeddings\n")
            f.write(f"Total Documents: {total_documents}\n")
            f.write("=" * 100 + "\n\n")
            
            # Write each document
            for i, (document, metadata) in enumerate(zip(
                results['documents'], 
                results['metadatas']
            )):
                
                # Get document ID if available
                doc_id = document_ids[i] if i < len(document_ids) else f"doc_{i+1}"
                
                # Document header
                f.write(f"DOCUMENT #{i+1}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Document ID: {doc_id}\n")
                f.write("-" * 40 + "\n")
                
                # Metadata section
                f.write("METADATA:\n")
                f.write(format_metadata(metadata))
                f.write("\n" + "-" * 40 + "\n")
                
                # Document content
                f.write("DOCUMENT CONTENT:\n")
                
                # Try to parse as JSON for pretty printing
                try:
                    parsed_json = json.loads(document)
                    f.write(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, write as plain text
                    f.write(document)
                
                f.write("\n\n" + "=" * 100 + "\n\n")
                
                # Progress indicator
                if (i + 1) % 10 == 0 or (i + 1) == total_documents:
                    print(f"  Progress: {i+1}/{total_documents} documents written")
        
        # Close ChromaDB connection
        chroma.close()
        
        print(f"✅ Export completed successfully!")
        print(f"📄 File saved: {output_path}")
        print(f"📊 Total documents exported: {total_documents}")
        
        # File size info
        file_size = output_path.stat().st_size
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        print(f"📏 File size: {size_str}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


def export_summary_statistics(chroma_handler: ChromaDBHandler) -> Dict[str, Any]:
    """Generate summary statistics about the ChromaDB collection."""
    try:
        collection = chroma_handler.client.get_collection("float_embeddings")
        results = collection.get(include=["metadatas"])
        
        stats = {
            "total_documents": len(results['metadatas']),
            "sources": {},
            "data_types": {},
            "measurement_availability": {
                "temperature": 0,
                "salinity": 0,
                "pressure": 0
            },
            "location_bounds": {
                "lat_min": float('inf'),
                "lat_max": float('-inf'),
                "lon_min": float('inf'),
                "lon_max": float('-inf')
            }
        }
        
        for metadata in results['metadatas']:
            # Count sources
            source = metadata.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
            
            # Count data types
            data_type = metadata.get('data_type', 'unknown')
            stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
            
            # Count measurements
            for measurement in ['temperature', 'salinity', 'pressure']:
                if metadata.get(f'has_{measurement}', False):
                    stats['measurement_availability'][measurement] += 1
            
            # Update location bounds
            for bound in ['lat_min', 'lat_max', 'lon_min', 'lon_max']:
                if bound in metadata and isinstance(metadata[bound], (int, float)):
                    value = float(metadata[bound])
                    if 'min' in bound:
                        stats['location_bounds'][bound] = min(stats['location_bounds'][bound], value)
                    else:
                        stats['location_bounds'][bound] = max(stats['location_bounds'][bound], value)
        
        return stats
        
    except Exception as e:
        print(f"Error generating statistics: {e}")
        return {}


def main():
    """Main function to run the export."""
    try:
        print("🚀 ChromaDB Data Export Tool")
        print("=" * 50)
        
        # Ask user for filename
        default_filename = f"chromadb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"\nDefault filename: {default_filename}")
        user_filename = input("Enter custom filename (or press Enter for default): ").strip()
        
        filename = user_filename if user_filename else default_filename
        
        # Ensure .txt extension
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        print(f"\n📄 Export filename: {filename}")
        
        # Perform export
        exported_file = export_chromadb_to_text(filename)
        
        if exported_file:
            print(f"\n🎉 Export completed successfully!")
            print(f"📁 File location: {exported_file}")
            
            # Ask if user wants to view summary statistics
            show_stats = input("\nWould you like to see summary statistics? (y/n): ").strip().lower()
            
            if show_stats in ['y', 'yes']:
                print("\n📊 Generating summary statistics...")
                chroma = ChromaDBHandler()
                stats = export_summary_statistics(chroma)
                chroma.close()
                
                if stats:
                    print("\n" + "=" * 50)
                    print("SUMMARY STATISTICS")
                    print("=" * 50)
                    print(f"Total Documents: {stats['total_documents']}")
                    
                    print(f"\nData Sources:")
                    for source, count in stats['sources'].items():
                        print(f"  {source}: {count}")
                    
                    print(f"\nData Types:")
                    for dtype, count in stats['data_types'].items():
                        print(f"  {dtype}: {count}")
                    
                    print(f"\nMeasurement Availability:")
                    for measurement, count in stats['measurement_availability'].items():
                        percentage = (count / stats['total_documents']) * 100
                        print(f"  {measurement.title()}: {count} ({percentage:.1f}%)")
                    
                    bounds = stats['location_bounds']
                    if bounds['lat_min'] != float('inf'):
                        print(f"\nLocation Coverage:")
                        print(f"  Latitude: {bounds['lat_min']:.3f}° to {bounds['lat_max']:.3f}°")
                        print(f"  Longitude: {bounds['lon_min']:.3f}° to {bounds['lon_max']:.3f}°")
        
        else:
            print("❌ Export failed or no data found")
        
    except KeyboardInterrupt:
        print("\n⚠️  Export cancelled by user")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")


if __name__ == "__main__":
    main()