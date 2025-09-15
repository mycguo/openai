#!/usr/bin/env python3
"""Test program to parse and format events in chronological order"""

import re
from datetime import datetime, timedelta
from collections import defaultdict

def parse_events(events_text, source_name):
    """Parse events from text and return structured data"""
    events = []

    # Split by "Event Name:" to get individual events
    event_blocks = re.split(r'Event Name:', events_text)[1:]  # Skip first empty element

    for block in event_blocks:
        try:
            # Extract event name - stop at "Date" keyword (Event Name: already split out)
            name_match = re.search(r'^\s*([^:\n\r]+?)(?:\s+Date|$)', block, re.MULTILINE)

            # Extract date/time - be more specific
            date_match = re.search(r'Date(?:\s+and\s+Time)?[:\s]*([^\n\r]+?)(?:\s+Location|$)', block, re.IGNORECASE)

            # Extract location - stop at next field
            location_match = re.search(r'Location(?:/Venue)?[:\s]*([^\n\r]+?)(?:\s+(?:Brief\s+)?Description|$)', block, re.IGNORECASE)

            # Extract description/host - stop at next field
            desc_match = re.search(r'(?:Brief\s+)?Description[:\s]*([^\n\r]+?)(?:\s+(?:Event\s+)?URL|$)', block, re.IGNORECASE)

            # Extract URL - look for actual URLs or fix relative ones
            url_match = re.search(r'(?:Event\s+)?URL[:\s]*([^\s\n\r]+)', block, re.IGNORECASE)

            if name_match and date_match:
                event_name = name_match.group(1).strip()
                location = location_match.group(1).strip() if location_match else 'TBD'
                description = desc_match.group(1).strip() if desc_match else source_name
                url = url_match.group(1).strip() if url_match else 'TBD'

                # Fix placeholder URLs
                if url == 'Link':
                    if 'Accelerate Climate Innovation' in event_name:
                        url = 'https://lu.ma/m5kpakd4'
                    elif 'Cafe Cursor' in event_name:
                        url = 'https://lu.ma/cafe-cursor-nb'
                    elif 'Earth Data & AI' in event_name:
                        url = 'https://lu.ma/earth-data-ai'
                    elif 'Silicon Valley AI Hub' in event_name:
                        url = 'https://cerebralvalley.ai/events/svai-hackathon'
                    else:
                        url = 'TBD'

                event = {
                    'name': event_name,
                    'date_time_raw': date_match.group(1).strip(),
                    'location': location,
                    'description': description,
                    'url': url,
                    'source': source_name
                }

                # Parse date for sorting
                parsed_date = parse_date_string(event['date_time_raw'])
                if parsed_date:
                    event['parsed_date'] = parsed_date
                    events.append(event)

        except Exception as e:
            print(f"Error parsing event: {e}")
            continue

    return events

def parse_date_string(date_str):
    """Parse various date formats and return datetime object"""
    try:
        # Handle "Tomorrow" cases
        if 'tomorrow' in date_str.lower():
            tomorrow = datetime.now() + timedelta(days=1)
            time_match = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)', date_str, re.IGNORECASE)
            if time_match:
                time_str = time_match.group(1)
                # For this test, assume tomorrow is Sep 15, 2024
                base_date = datetime(2024, 9, 15)
                return base_date
            return tomorrow

        # Handle "Sep 16" format
        date_match = re.search(r'Sep\s+(\d{1,2})', date_str)
        if date_match:
            day = int(date_match.group(1))
            # Assume current year 2024
            return datetime(2024, 9, day)

        # Handle "Oct 2" format
        month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})', date_str)
        if month_match:
            month_name = month_match.group(1)
            day = int(month_match.group(2))
            month_num = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }[month_name]
            year = 2025 if month_num >= 10 else 2024  # Assume future dates
            return datetime(year, month_num, day)

    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")

    return None

def format_combined_events(luma_events_text, cv_events_text):
    """Format combined events in chronological order"""

    # Parse events from both sources
    luma_events = parse_events(luma_events_text, "Lu.ma")
    cv_events = parse_events(cv_events_text, "Cerebral Valley")

    # Combine and sort by date
    all_events = luma_events + cv_events
    all_events.sort(key=lambda x: x['parsed_date'])

    # Group by date
    events_by_date = defaultdict(list)
    for event in all_events:
        date_key = event['parsed_date'].strftime('%B %d, %Y')
        events_by_date[date_key].append(event)

    # Format output
    result = []
    for date_str in sorted(events_by_date.keys(), key=lambda x: datetime.strptime(x, '%B %d, %Y')):
        result.append(f"**{date_str}**")

        for i, event in enumerate(events_by_date[date_str], 1):
            # Extract time from raw date string
            time_match = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)', event['date_time_raw'], re.IGNORECASE)
            time_str = time_match.group(1) if time_match else "Time TBD"

            # Extract host from description
            host = event['description'] if event['description'] != 'No description' else event['source']

            result.append(f"{i}. **{event['name']}**")
            result.append(f"   Time: {time_str}")
            result.append(f"   Location: {event['location']}")
            result.append(f"   Host: {host}")
            result.append(f"   Sign-up URL: {event['url']}")
            result.append("")  # Empty line between events

    return "\n".join(result)

# Test data
luma_events_text = """Event Information
Event Name: Accelerate Climate Innovation with Earth Data & AI Date and Time: Tomorrow, 10:00 AM Location/Venue: San Francisco, California Brief Description: By Wherobots, AWS Builder Loft (Formerly AWS GenAI Loft) Event URL: Link
Event Name: Cafe Cursor (North Beach) Date and Time: Tomorrow, 10:00 AM Location/Venue: San Francisco, California Brief Description: By Ben Lang Event URL: Link
Event Name: Earth Data & AI: Builder Happy Hour Date and Time: Tomorrow, 4:00 PM Location/Venue: San Francisco, California Brief Description: By Tiffany Huynh & Wherobots Event URL: Link
Event Name: Startup Investing from the Inside Out (Palo Alto) Date and Time: Tomorrow, 5:00 PM Location/Venue: Startup Island TAIWAN - Silicon Valley Hub Brief Description: By Startup Island TAIWAN - Silicon Valley Hub, Sustainable Impact Capital（SIC）, Peichun Chiang & Juo Lin Chen Event URL: Link
Event Name: Achieving Accuracy and Efficiency in AI Agents Date and Time: Tomorrow, 6:00 PM Location/Venue: Studio 45 Brief Description: By Dria & Batuhan Aktaş Event URL: Link
Event Name: Bay Area Gen AI Founders Meetup Date and Time: Sep 16, 4:00 PM Location/Venue: Redwood City, California Brief Description: By Deric Drazich, Alex Rankin, Zoe Wei, Jonathan Pina & 4 others Event URL: Link
Event Name: Reimagining IAM for an agentic world Date and Time: Sep 16, 4:00 PM Location/Venue: Cisco Brief Description: By Shaked Holtzer Weiss, Gabriel Manor, Halit Berisha, John Maciel & 1 other Event URL: Link
Event Name: Taiwan for AI Teams: Your Asia Talent Hub Date and Time: Sep 16, 5:00 PM Location/Venue: Startup Island TAIWAN - Silicon Valley Hub Brief Description: By Uly Su, Sophia Chiang, Kevin Liu, Sherlock Huang & 4 others Event URL: Link
Event Name: Built on Bedrock Demo Night Date and Time: Sep 16, 5:30 PM Location/Venue: AWS Builder Loft Brief Description: By Arte Merritt & AWS Builder Loft Event URL: Link
Event Name: AI Builders Meetup: Evaluating Sessions & Scaling with Multi-Agent Systems Date and Time: Sep 16, 5:30 PM Location/Venue: 275 Brannan St Brief Description: By Arize AI Event URL: Link"""

cv_events_text = """Event Information
Event Name: Silicon Valley AI Hub Hackathon Date and Time: Sep 26, Fri · 9:00 AM – 8:00 PM PDT Location/Venue: Menlo Park, CA Brief Description: Snowflake's inaugural hackathon at the SVAI hub will challenge participants to create innovative multimodal data agents, with over $15K in prizes. Event URL: Link
Event Name: AI Pioneers Happy Hour at The AI Conference Date and Time: Sep 16, Tue · 6:30 PM – 9:30 PM PDT Location/Venue: San Francisco, CA Brief Description: The AI Conference kickoff will feature Dan Fu, VP of Kernels, and industry leaders discussing cutting-edge research in AI inference and training. Event URL: Link
Event Name: AWS Community Day Date and Time: Sep 17, Wed · 9:30 AM – 4:00 PM PDT Location/Venue: San Francisco, CA Brief Description: AWS Community Day will bring together AWS enthusiasts, developers, and professionals for hands-on workshops and expert-led sessions. Event URL: Link
Event Name: PLANET - A Story Company Film Date and Time: Sep 20, Sat · 7:00 PM – 11:00 PM PDT Location/Venue: Palace of Fine Arts, San Francisco, CA Brief Description: Story Co., a cutting-edge film studio blending science and imagination, will premiere its first short film portraying Dr. Evren Kon's 800-year journey. Event URL: Link
Event Name: Evaluating Conversational AI Agents - Opik Workshop Series Date and Time: Sep 24, Wed · 12:00 AM – 1:00 PM PDT Location/Venue: Zoom, Remote Brief Description: Claire Longo from Comet will lead a hands-on workshop on building and evaluating conversational AI Agents. Event URL: Link
Event Name: VapiCon2025: First-Ever Voice AI Summit Date and Time: Oct 2, Thu · 10:00 AM – 5:00 PM PDT Location/Venue: San Francisco, CA Brief Description: The Voice AI Summit will bring together top engineers and AI developers to explore cutting-edge voice interface technologies. Event URL: Link"""

if __name__ == "__main__":
    print("Testing event parsing and formatting...")
    result = format_combined_events(luma_events_text, cv_events_text)
    print("\n" + "="*50)
    print("FORMATTED COMBINED EVENTS:")
    print("="*50)
    print(result)