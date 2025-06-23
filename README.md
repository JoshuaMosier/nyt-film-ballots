# NYT Film Ballots Data Extractor

## Overview
This project extracts structured data from The New York Times' "Best Movies of the 21st Century" celebrity ballots HTML page. The extraction captures comprehensive information about film industry professionals' movie preferences, survey metadata, and detailed movie information.

## Data Source
- **Source**: NYT interactive article featuring celebrity ballots
- **Content**: 119 film industry professionals (actors, directors, writers, etc.) sharing their top movie picks
- **Format**: Large HTML file (2.8MB) with embedded JavaScript data structures

## Technical Approach

### Challenge
The HTML file contains structured data embedded in JavaScript sections rather than in the DOM. Initial attempts to parse HTML elements would have been incomplete and error-prone.

### Solution
The script identifies and extracts JavaScript data structures, specifically:
1. **Main data structure**: `data: [null, { "type": "data", "data": { ... }}]`
2. **People array**: Contains all ballot information
3. **Survey metadata**: Article information, publication dates, bylines

### Key Technical Details
- **JavaScript to JSON conversion**: Converts JS object syntax to valid JSON
- **Robust bracket matching**: Handles nested objects/arrays with proper string escaping
- **Fallback extraction**: Multiple patterns to ensure data capture
- **Error handling**: Debug files saved when parsing fails

## Data Structure

### Extracted Information
```
{
  "people": [119 professionals with movie picks],
  "survey_metadata": {
    "headline": "Article title",
    "first_published": "June 22, 2025",
    "last_modified": "June 23, 2025",
    "bylines": [author information]
  },
  "theme": "news",
  "extraction_stats": {
    "total_people": 119,
    "total_movies": 1181
  }
}
```

### Person Data Structure
```json
{
  "slug": "person-identifier",
  "name": "Full Name",
  "bio": "Professional background",
  "categories": ["actor", "director", etc.],
  "top_choice": "Highlighted pick",
  "notes": "Additional comments",
  "movies": [
    {
      "rank": 1,
      "title": "Movie Title",
      "year": "2001",
      "review_url": "https://nytimes.com/...",
      "nyt_id": "unique_nyt_identifier",
      "imdb_id": "tt1234567"
    }
  ]
}
```

## Usage

### Basic Extraction
```bash
python extract_nyt_film_data.py page.html
```

### With Statistics
```bash
python extract_nyt_film_data.py page.html --stats
```

### Output Formats
- `--format json` - JSON only
- `--format csv` - CSV files only  
- `--format both` - Both formats (default)

## Output Files

### Generated Files
1. **`nyt_film_data.json`** (431KB) - Complete structured data
2. **`nyt_film_data_people.csv`** (121 rows) - People information
3. **`nyt_film_data_movies.csv`** (1,183 rows) - All movie selections
4. **`nyt_film_data_metadata.csv`** (8 rows) - Survey metadata
5. **`nyt_film_data_stats.json`** - Detailed analytics

### Key Statistics
- **119 people** across 8 professional categories
- **1,181 total movie selections**
- **482 unique movies**
- **Most popular**: Mulholland Drive (27 selections)
- **Peak year**: 2001 (87 movie selections)

## Dependencies
```
beautifulsoup4>=4.12.0  # Not used in final version but kept for compatibility
lxml>=4.9.0            # Not used in final version but kept for compatibility
```

Note: The final implementation uses only Python standard library (re, json, csv) for maximum compatibility.

## Code Architecture

### Main Functions
- `extract_javascript_data()` - Finds and extracts JS data structures
- `convert_js_to_json()` - Converts JavaScript object syntax to JSON
- `extract_complete_data_object()` - Handles nested bracket matching
- `process_person_data()` - Normalizes person/movie data
- `generate_summary_stats()` - Creates analytics

### Error Handling
- Debug files saved when JSON parsing fails
- Multiple extraction patterns as fallbacks
- Comprehensive logging throughout process

## Use Cases for Extracted Data

### Analysis Opportunities
- **Film preference trends** by professional category
- **Movie popularity rankings** across industry professionals
- **Year-based analysis** of cinema preferences
- **Professional network analysis** through shared preferences
- **Review URL mapping** for deeper film analysis

### Data Quality
- **Complete IMDB/NYT IDs** for all movies enable external API integration
- **Professional categories** allow for demographic analysis
- **Temporal data** (years, publication dates) enable trend analysis
- **Review URLs** provide direct links to NYT film criticism

## Notes for AI Implementation

### Key Insights
1. **HTML parsing insufficient** - Structured data was in JavaScript, not DOM
2. **JavaScript ≠ JSON** - Required careful syntax conversion
3. **Nested data structures** - Main data contained multiple sub-objects
4. **Robust extraction needed** - Large file size required efficient parsing
5. **Multiple output formats** - CSV and JSON serve different analysis needs

### Potential Extensions
- Add movie genre classification via IMDB API
- Cross-reference with box office data
- Sentiment analysis of professional bios
- Network analysis of shared movie preferences
- Time-series analysis of publication patterns

This extraction approach can be adapted for similar embedded JavaScript data scenarios in other HTML sources. 