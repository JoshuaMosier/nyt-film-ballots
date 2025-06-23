"""
NYT Film Ballots Data Extractor
Extracts data from HTML file containing NYT's "Best Movies of 21st Century" celebrity ballots
Now extracts from embedded JavaScript JSON data for more accurate results
"""

import re
import json
import csv
from pathlib import Path
import argparse
from typing import List, Dict, Any

def convert_js_to_json(js_string: str) -> str:
    """Convert JavaScript object syntax to valid JSON"""
    
    # Split by lines to process more carefully
    lines = js_string.split('\n')
    result_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            result_lines.append(line)
            continue
            
        # Handle property definitions: propertyName: value
        # Match lines with property definitions
        prop_match = re.match(r'^(\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*(.*)$', line)
        if prop_match:
            indent, prop_name, value = prop_match.groups()
            # Add quotes around property name
            new_line = f'{indent}"{prop_name}": {value}'
            result_lines.append(new_line)
        else:
            result_lines.append(line)
    
    js_string = '\n'.join(result_lines)
    
    # Fix common JavaScript vs JSON differences
    js_string = js_string.replace('undefined', 'null')
    js_string = js_string.replace('void 0', 'null')
    
    return js_string

def extract_complete_data_object(html_content: str, start_pos: int) -> str:
    """Extract a complete JavaScript object/array starting from a position"""
    
    # Find the start of the object/array
    brace_pos = start_pos
    while brace_pos < len(html_content) and html_content[brace_pos] not in '{[':
        brace_pos += 1
    
    if brace_pos >= len(html_content):
        return None
    
    start_char = html_content[brace_pos]
    end_char = '}' if start_char == '{' else ']'
    
    # Find the matching closing brace/bracket
    bracket_count = 0
    current_pos = brace_pos
    in_string = False
    escape_next = False
    
    while current_pos < len(html_content):
        char = html_content[current_pos]
        
        if escape_next:
            escape_next = False
        elif char == '\\' and in_string:
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if char in '{[':
                bracket_count += 1
            elif char in '}]':
                bracket_count -= 1
                if bracket_count == 0:
                    # Found the closing bracket
                    end_pos = current_pos
                    return html_content[brace_pos:end_pos + 1]
        
        current_pos += 1
    
    return None

def extract_javascript_data(html_content: str) -> Dict[str, Any]:
    """Extract all structured data from the JavaScript sections"""
    
    all_data = {}
    
    # Look for the main data structure: data: [null, { "type": "data", "data": { ... }}]
    data_pattern = r'data:\s*\[null,\s*\{'
    data_match = re.search(data_pattern, html_content)
    
    if data_match:
        print(f"Found main data structure at position {data_match.start()}")
        
        # Extract the complete data object
        main_data_js = extract_complete_data_object(html_content, data_match.start())
        if main_data_js:
            # Extract just the data part after "data: "
            data_start = main_data_js.find('[')
            if data_start != -1:
                main_data_js = main_data_js[data_start:]
                
                print(f"Extracted main data structure, length: {len(main_data_js)}")
                
                # Convert to JSON and parse
                main_data_json = convert_js_to_json(main_data_js)
                
                try:
                    main_data = json.loads(main_data_json)
                    print(f"Successfully parsed main data structure")
                    
                    # Extract the nested data object
                    if len(main_data) >= 2 and isinstance(main_data[1], dict):
                        nested_data = main_data[1].get('data', {})
                        
                        # Extract survey metadata
                        if 'body' in nested_data and isinstance(nested_data['body'], list):
                            for item in nested_data['body']:
                                if isinstance(item, dict) and item.get('type') == 'svelte':
                                    value = item.get('value', {})
                                    if 'header' in value:
                                        all_data['survey_metadata'] = {
                                            'headline': value['header'].get('headline', ''),
                                            'custom_headline': value['header'].get('customHeadline', ''),
                                            'kicker': value['header'].get('kicker', ''),
                                            'first_published': value['header'].get('firstPublished', ''),
                                            'last_modified': value['header'].get('lastModified', ''),
                                            'bylines': value['header'].get('bylines', [])
                                        }
                                        print("Extracted survey metadata")
                        
                        # Extract people data
                        if 'people' in nested_data:
                            all_data['people'] = nested_data['people']
                            print(f"Extracted {len(nested_data['people'])} people from main data")
                        
                        # Extract theme and other metadata
                        if 'theme' in nested_data:
                            all_data['theme'] = nested_data['theme']
                        
                except json.JSONDecodeError as e:
                    print(f"Failed to parse main data structure: {e}")
                    # Save for debugging
                    with open('debug_main_data.json', 'w', encoding='utf-8') as f:
                        f.write(main_data_json)
                    print("Saved problematic main data JSON to debug_main_data.json")
    
    # Fallback: Look for people array directly if not found in main structure
    if 'people' not in all_data:
        people_pos = html_content.find('people:')
        if people_pos != -1:
            print(f"Fallback: Found 'people:' at position {people_pos}")
            
            # Extract people array
            people_js = extract_complete_data_object(html_content, people_pos + 7)  # Skip "people:"
            if people_js:
                print(f"Extracted people array, length: {len(people_js)}")
                
                # Convert JavaScript object syntax to JSON
                people_json = convert_js_to_json(people_js)
                
                try:
                    people_data = json.loads(people_json)
                    all_data['people'] = people_data
                    print(f"Successfully parsed {len(people_data)} people via fallback")
                except json.JSONDecodeError as e:
                    print(f"Fallback people parsing failed: {e}")
                    with open('debug_people.json', 'w', encoding='utf-8') as f:
                        f.write(people_json)
                    print("Saved problematic people JSON to debug_people.json")
    
    return all_data if all_data else None

def process_person_data(person: Dict[str, Any]) -> Dict[str, Any]:
    """Process and clean person data from the JavaScript structure"""
    
    processed = {
        'slug': person.get('slug', ''),
        'name': person.get('name', ''),
        'bio': person.get('bio', ''),
        'categories': person.get('categories', []),
        'top_choice': person.get('top', ''),
        'notes': person.get('notes', ''),
        'movies': []
    }
    
    # Process movie choices
    choices = person.get('choices', [])
    for i, choice in enumerate(choices, 1):
        movie_data = {
            'rank': i,
            'title': choice.get('title', ''),
            'year': choice.get('year', ''),
            'review_url': choice.get('review', ''),
            'nyt_id': choice.get('nyt_id', ''),
            'imdb_id': choice.get('imdb_id', ''),
            'person_slug': choice.get('slug', '')
        }
        processed['movies'].append(movie_data)
    
    return processed

def extract_all_data(html_file_path: str) -> Dict[str, Any]:
    """Extract all data from the JavaScript section in the HTML file"""
    print(f"Reading HTML file: {html_file_path}")
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract JavaScript data
    js_data = extract_javascript_data(content)
    if not js_data or 'people' not in js_data:
        print("No data found in JavaScript section")
        return {'people': [], 'metadata': {}}
    
    people_raw = js_data['people']
    print(f"Found {len(people_raw)} people in JavaScript data")
    
    processed_people = []
    for i, person in enumerate(people_raw, 1):
        print(f"Processing person {i}/{len(people_raw)}: {person.get('name', 'Unknown')}")
        processed_person = process_person_data(person)
        processed_people.append(processed_person)
    
    # Combine all extracted data
    result = {
        'people': processed_people,
        'survey_metadata': js_data.get('survey_metadata', {}),
        'theme': js_data.get('theme', ''),
        'extraction_stats': {
            'total_people': len(processed_people),
            'total_movies': sum(len(person['movies']) for person in processed_people)
        }
    }
    
    return result

def save_to_json(data: Dict[str, Any], output_file: str):
    """Save data to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"Data saved to JSON: {output_file}")

def save_to_csv(data: Dict[str, Any], output_file: str):
    """Save data to CSV files with separate files for people and movies"""
    
    people_data = data.get('people', [])
    
    # Save people data
    people_file = output_file.replace('.csv', '_people.csv')
    with open(people_file, 'w', newline='', encoding='utf-8') as file:
        if people_data:
            fieldnames = ['slug', 'name', 'bio', 'categories', 'movie_count', 'top_choice', 'notes']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            for person in people_data:
                writer.writerow({
                    'slug': person['slug'],
                    'name': person['name'],
                    'bio': person['bio'],
                    'categories': '; '.join(person['categories']),
                    'movie_count': len(person['movies']),
                    'top_choice': person['top_choice'],
                    'notes': person['notes']
                })
    
    # Save movies data
    movies_file = output_file.replace('.csv', '_movies.csv')
    with open(movies_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['person_slug', 'person_name', 'rank', 'title', 'year', 'review_url', 'nyt_id', 'imdb_id']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for person in people_data:
            for movie in person['movies']:
                writer.writerow({
                    'person_slug': person['slug'],
                    'person_name': person['name'],
                    'rank': movie['rank'],
                    'title': movie['title'],
                    'year': movie['year'],
                    'review_url': movie['review_url'],
                    'nyt_id': movie['nyt_id'],
                    'imdb_id': movie['imdb_id']
                })
    
    # Save metadata
    metadata_file = output_file.replace('.csv', '_metadata.csv')
    with open(metadata_file, 'w', newline='', encoding='utf-8') as file:
        metadata = data.get('survey_metadata', {})
        if metadata:
            writer = csv.writer(file)
            writer.writerow(['Field', 'Value'])
            for key, value in metadata.items():
                if isinstance(value, list):
                    value = '; '.join(str(v) for v in value)
                writer.writerow([key, str(value)])
    
    print(f"Data saved to CSV files: {people_file}, {movies_file}, and {metadata_file}")

def generate_summary_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics about the data"""
    people_data = data.get('people', [])
    total_people = len(people_data)
    total_movies = sum(len(person['movies']) for person in people_data)
    
    # Count unique movies
    unique_movies = set()
    movie_counts = {}
    year_counts = {}
    
    for person in people_data:
        for movie in person['movies']:
            title = movie['title']
            year = movie['year']
            if title:
                unique_movies.add(title)
                movie_counts[title] = movie_counts.get(title, 0) + 1
                if year:
                    year_counts[year] = year_counts.get(year, 0) + 1
    
    # Most popular movies
    most_popular = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Most popular years
    most_popular_years = sorted(year_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # People with most movies
    people_movie_counts = [(person['name'], len(person['movies'])) for person in people_data]
    people_movie_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Category breakdown
    category_counts = {}
    for person in people_data:
        for category in person['categories']:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    stats = {
        'total_people': total_people,
        'total_movie_selections': total_movies,
        'unique_movies': len(unique_movies),
        'average_movies_per_person': round(total_movies / total_people, 2) if total_people > 0 else 0,
        'most_popular_movies': most_popular,
        'most_popular_years': most_popular_years,
        'people_with_most_movies': people_movie_counts[:10],
        'category_breakdown': sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    }
    
    # Add survey metadata to stats if available
    if 'survey_metadata' in data:
        stats['survey_info'] = data['survey_metadata']
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Extract data from NYT film ballots HTML file')
    parser.add_argument('html_file', help='Path to the HTML file')
    parser.add_argument('--output', '-o', default='nyt_film_data', help='Output file prefix (default: nyt_film_data)')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'both'], default='both', help='Output format (default: both)')
    parser.add_argument('--stats', '-s', action='store_true', help='Generate summary statistics')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.html_file).exists():
        print(f"Error: HTML file '{args.html_file}' not found")
        return
    
    # Extract data
    data = extract_all_data(args.html_file)
    
    if not data or not data.get('people'):
        print("No data extracted!")
        return
    
    # Save data in requested formats
    if args.format in ['json', 'both']:
        save_to_json(data, f"{args.output}.json")
    
    if args.format in ['csv', 'both']:
        save_to_csv(data, f"{args.output}.csv")
    
    # Generate statistics if requested
    if args.stats:
        stats = generate_summary_stats(data)
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total people: {stats['total_people']}")
        print(f"Total movie selections: {stats['total_movie_selections']}")
        print(f"Unique movies: {stats['unique_movies']}")
        print(f"Average movies per person: {stats['average_movies_per_person']}")
        
        print(f"\nMost popular movies:")
        for movie, count in stats['most_popular_movies']:
            print(f"  {movie}: {count} selections")
        
        print(f"\nMost popular years:")
        for year, count in stats['most_popular_years']:
            print(f"  {year}: {count} movies")
        
        print(f"\nPeople with most movie selections:")
        for name, count in stats['people_with_most_movies']:
            print(f"  {name}: {count} movies")
        
        print(f"\nCategory breakdown:")
        for category, count in stats['category_breakdown']:
            print(f"  {category}: {count} people")
        
        # Save stats to JSON
        stats_file = f"{args.output}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as file:
            json.dump(stats, file, indent=2, ensure_ascii=False)
        print(f"\nDetailed statistics saved to: {stats_file}")
    
    print(f"\nExtraction complete! Processed {len(data.get('people', []))} people.")
    
    # Print survey metadata if available
    if data.get('survey_metadata'):
        metadata = data['survey_metadata']
        print(f"\nSurvey Information:")
        if metadata.get('headline'):
            print(f"  Headline: {metadata['headline']}")
        if metadata.get('first_published'):
            print(f"  Published: {metadata['first_published']}")
        if metadata.get('last_modified'):
            print(f"  Last Modified: {metadata['last_modified']}")

if __name__ == "__main__":
    main() 