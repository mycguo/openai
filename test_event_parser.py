#!/usr/bin/env python3
"""Test program to parse and format events in chronological order using LLM"""

import os
import sys
from openai import OpenAI

# Add parent directory to path to import from events.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from events import generate_from_openai

def format_combined_events_with_llm(luma_events_text, cv_events_text):
    """Use LLM to combine and format events in chronological order"""

    # Combine all events into a single text
    combined_text = f"""
Lu.ma Events:
{luma_events_text}

Cerebral Valley Events:
{cv_events_text}
"""

    # Use OpenAI to combine and format the events
    prompt = """Take all the events from both sources and combine them into a single chronologically ordered list.
Format the output EXACTLY like this (each field on its own line):

**[Date in format: Month DD, YYYY]**

1. **[Event Name]**
   Time: [Time or "Time TBD"]
   Location: [Location]
   Host: [Host organization or description]
   Sign-up URL: [Show the actual URL directly, not "Link". If no URL available, show "TBD"]

2. **[Next Event Name]**
   Time: [Time]
   Location: [Location]
   Host: [Host]
   Sign-up URL: [URL]

IMPORTANT FORMATTING RULES:
- Each event field (Time, Location, Host, Sign-up URL) MUST be on its own line
- Add two spaces at the end of each line to create proper line breaks in markdown
- Leave a blank line between events for readability
- Group all events by date, sort dates chronologically
- Combine events from BOTH sources into single date groups (don't separate by source)
- When you see "Event URL: Link", extract the actual URL
- If the URL shows as just "Link" with no actual URL, display "TBD" instead

IMPORTANT:
- Combine ALL events from both sources into a single unified list, not two separate sections.
- Show actual URLs directly (e.g., https://lu.ma/event-name), never just show "Link"
- If a URL appears as "[text](url)" markdown format, extract and show just the URL"""

    try:
        result = generate_from_openai(combined_text + "\n\n" + prompt, temperature=0.1, max_tokens=3000)
        return result
    except Exception as e:
        return f"Error combining events: {str(e)}"

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
    print("Testing event parsing and formatting with LLM...")
    result = format_combined_events_with_llm(luma_events_text, cv_events_text)
    print("\n" + "="*50)
    print("FORMATTED COMBINED EVENTS (LLM):")
    print("="*50)
    print(result)