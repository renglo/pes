# Seed Cases for Database Insertion

This directory contains seed cases that can be inserted into the database for the plan generation system.

## Files

- **`seed_cases.json`** - Database-ready format (use this for insertion)
- **`seed_cases_readable.json`** - Human-readable format (for reference/editing)

## Database Structure

Each case in the database should have the following structure:

```json
{
  "id": "unique_case_id",
  "text": "{\"signature\":\"{...}\",\"plan\":{\"steps\":[...]}}",
  "meta": {
    "destination": "...",
    "case_type": "...",
    "season": "...",
    "purpose": "..."
  }
}
```

### Field Descriptions

- **`id`**: Unique identifier for the case (e.g., "seed_case_001")
- **`text`**: JSON string containing:
  - `signature`: A JSON string (compact format, as returned by `Signature.to_text()`)
  - `plan`: An object with a `steps` array containing plan steps
- **`meta`**: Metadata object for filtering and searching cases

## How to Insert into Database

The cases should be inserted into the `pes_cases` ring (or your configured case ring) in the database.

### Example Insertion Code

```python
import json
from your_data_controller import DataController

# Load the seed cases
with open('seed_cases.json', 'r') as f:
    cases = json.load(f)

# Insert each case
for case in cases:
    # The 'text' field is already a JSON string, so use it directly
    data_controller.create_or_update(
        portfolio="your_portfolio",
        org="your_org",
        ring="pes_cases",
        item_id=case['id'],
        text=case['text'],
        meta=case['meta']
    )
```

## Case Descriptions

### Case 1: NYC Business Conference (Single Destination)
- **Type**: Single destination trip
- **Season**: Summer
- **Purpose**: Conference attendance
- **Activities**: Conference, bike rental
- **Origin**: Boston → NYC → Boston

### Case 2: Multi-City Business Trip
- **Type**: Multi-city trip
- **Season**: Winter
- **Purpose**: Client meetings
- **Activities**: Meetings in NYC and DC
- **Route**: Chicago → NYC → Washington DC → Chicago

### Case 3: San Francisco Tech Conference
- **Type**: Single destination trip
- **Season**: Spring
- **Purpose**: Tech conference
- **Activities**: Conference, bike rental, route planning
- **Origin**: Seattle → San Francisco → Seattle

### Case 4: Day Trip to Lima
- **Type**: Day trip (same-day return)
- **Season**: All year
- **Purpose**: Client meeting
- **Activities**: Business meeting
- **Route**: Sao Paulo → Lima → Sao Paulo (same day)

## Notes

- The `signature` field in the `text` JSON string is itself a JSON string (double-encoded)
- This matches the format used by `Signature.to_text()` which returns a compact JSON string
- The `plan.steps` array contains all the plan steps with their actions, inputs, and dependencies
- Metadata fields can be used for filtering cases by destination, type, season, or purpose

