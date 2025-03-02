import streamlit as st
import pandas as pd
import requests
import os
from gtts import gTTS
from gtts.lang import tts_langs
from io import BytesIO
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptAvailable
from googleapiclient.discovery import build
from datetime import datetime, timedelta

# ---------------------
#   API KEYS
# ---------------------
# Replace these with your actual keys or retrieve them from st.secrets
YOUTUBE_API_KEY = ""
CHATGPT_API_KEY = ""
CHATGPT_API_URL = ""

# -----------------------------------------------------------------
#  1. INITIALIZE CSV FILES IF NOT PRESENT (CHANNELS.CSV AND VIDEOS.CSV)
# -----------------------------------------------------------------
def initialize_csv_files():
    os.makedirs('data', exist_ok=True)

    # channels.csv: channel_id, channel_url, channel_title, description, topics
    if not os.path.exists('data/channels.csv'):
        channels_df = pd.DataFrame(columns=[
            'channel_id', 'channel_url', 'channel_title', 'description', 'topics'
        ])
        channels_df.to_csv('data/channels.csv', index=False, sep='|')

    # videos.csv: video_id, video_title, channel_id, transcript, language_code, short_summary, detailed_summary
    if not os.path.exists('data/videos.csv'):
        videos_df = pd.DataFrame(columns=[
            'video_id', 'video_title', 'channel_id', 'transcript', 'language_code', 'short_summary', 'detailed_summary'
        ])
        videos_df.to_csv('data/videos.csv', index=False, sep='|')


# -----------------------------------------------------------------
#  2. LOAD DATA
# -----------------------------------------------------------------
def load_channels():
    if os.path.exists('data/channels.csv'):
        return pd.read_csv('data/channels.csv', sep='|')
    else:
        return pd.DataFrame(columns=[
            'channel_id', 'channel_url', 'channel_title', 'description', 'topics'
        ])

def load_videos():
    if os.path.exists('data/videos.csv'):
        return pd.read_csv('data/videos.csv', sep='|')
    else:
        return pd.DataFrame(columns=[
            'video_id', 'video_title', 'channel_id', 'transcript', 'language_code', 'short_summary', 'detailed_summary'
        ])


# -----------------------------------------------------------------
#  3. UTILITY FUNCTIONS
# -----------------------------------------------------------------
def extract_channel_id(url: str, api_key: str) -> str:
    """
    Extracts the channel ID from various YouTube URL formats, including '@' handles.
    """
    patterns = [
        r"youtube\.com/channel/([A-Za-z0-9_-]{24})",  # e.g. youtube.com/channel/UCxxxxxx
        r"youtube\.com/c/([A-Za-z0-9_-]+)",           # e.g. youtube.com/c/SomeChannel
        r"youtube\.com/user/([A-Za-z0-9_-]+)",        # e.g. youtube.com/user/SomeUser
        r"youtube\.com/@([A-Za-z0-9_-]+)"             # e.g. youtube.com/@ChannelHandle
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            # If it's an '@' handle, attempt to resolve via YT Data API
            if "@" in url:
                return resolve_channel_id(api_key, match.group(1))
            return match.group(1)
    return None

def resolve_channel_id(api_key: str, handle: str) -> str:
    """
    Resolve a YouTube channel '@' handle to a channel ID using the YouTube Data API.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        request = youtube.search().list(
            part="snippet",
            q=handle,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        items = response.get('items', [])
        if items:
            return items[0]['snippet']['channelId']
    except Exception as e:
        print(f"Error resolving handle '{handle}': {e}")
    return None

def get_channel_description(api_key: str, channel_id: str) -> str:
    """
    Retrieve channel description via YouTube Data API.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.channels().list(part='snippet', id=channel_id)
    response = request.execute()
    items = response.get('items', [])
    if items:
        return items[0]['snippet']['description']
    return ""

def categorize_channel(description: str, api_url: str, api_key: str) -> list:
    """
    Use ChatGPT to categorize the channel into 1 or 2 broad topics 
    (e.g., Science, Politics, Funny, Technology, Education, Entertainment, Sports, Other).
    """
    prompt = (
        "You are given a YouTube channel description. "
        "Select ONLY ONE or TWO most relevant generic topics from this list: "
        "[Science, Politics, Funny, Technology, Education, Entertainment, Sports, Other]. "
        "Return them as a comma-separated list (e.g., 'Technology' or 'Technology, Science'). "
        "If no suitable category is found, return 'Other'.\n\n"
        f"Description:\n{description}"
    )

    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }

    data = {
        "messages": [
            {"role": "system", "content": "You are useful system used to extract topics from video description."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 60,
        "n": 1
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        topics_text = result['choices'][0]['message']['content'].strip()
        # Split and clean topics
        topics_list = [t.strip().capitalize() for t in topics_text.split(',')]
        return topics_list
    else:
        st.error(f"Error categorizing channel: {response.text}")
        return []

def get_latest_videos(api_key: str, channel_id: str, days: int = 7) -> list:
    """
    Fetch up to 3 videos from a channel published in the last 'days' days (default 7).
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    one_week_ago = datetime.utcnow() - timedelta(days=days)
    published_after = one_week_ago.isoformat("T") + "Z"

    videos = []
    next_page_token = None

    while True:
        request = youtube.search().list(
            part='id,snippet',
            channelId=channel_id,
            publishedAfter=published_after,
            maxResults=50,
            order='date',
            type='video',
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get('items', []):
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            published_at = item['snippet']['publishedAt']
            channel_title = item['snippet']['channelTitle']
            videos.append({
                'video_id': video_id,
                'title': title,
                'published_at': published_at,
                'channel_id': channel_id,
                'channel_title': channel_title
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Sort videos by published date descending (latest first)
    videos.sort(key=lambda v: v['published_at'], reverse=True)
    # Return only the first 3
    return videos[:3]

# Define a function to remove stopwords and articles
def remove_stopwords_and_articles(text):
    """
    1) If 'text' is more than 2,200 words, keep only the first 2,000 words 
       and the last 200 words, and remove everything in between.
    2) Remove articles and common stopwords to further reduce token count.
    """
    
    # List of articles and common stopwords to remove
    articles_and_stopwords = {
        'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet',
        'to', 'of', 'at', 'by', 'from', 'with', 'about', 'as', 'this',
        'is', 'am', 'are',
        'be', 'been', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',

        'bir', 've', 'veya', 'ama', 'fakat', 'ancak', 'ile', 'da', 'de', 
        'bu', 'şu', 'o', 'şunlar', 'onlar', 'şuraya', 'orada', 'hangi', 
        'kim', 'ne', 'neden', 'nasıl', 'değil', 'daha', 'çok', 'çokça', 
        'hiç', 'hep', 'her', 'bazı', 'kimi', 'şimdi', 'sonra', 'önce', 
        'işte', 'evet', 'hayır', 'var', 'yok', 'olarak', 'olarak', 'üzere', 
        'zaten', 'dolayı', 'nedeniyle', 'özellikle', 'herhangi', 
        'niye', 'niçin', 'şey', 'şeyi', 'şeyler', 'ya'
    }
    
    # Split the text into words
    words = text.split()
    
    # If the transcript has more than 2200 words, 
    # keep only the first 2000 and the last 200.
    if len(words) > 2200:
        words = words[:2000] + words[-200:]
    
    # Filter out the articles and stopwords
    filtered_words = [word for word in words if word.lower() not in articles_and_stopwords]
    
    # Return the processed text
    return ' '.join(filtered_words)


def get_transcript(video_id: str) -> (str, str):
    """
    Retrieve the transcript in any available language.
    Returns a tuple of (transcript_text, language_code).
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Iterate over transcripts; prefer manually created transcripts
        for transcript in transcript_list:
            if not transcript.is_generated:
                try:
                    fetched_transcript = transcript.fetch()
                    full_text = " ".join([entry['text'] for entry in fetched_transcript])
                    return full_text, transcript.language_code
                except Exception:
                    continue

        # If no manually created transcript, try auto-generated
        for transcript in transcript_list:
            if transcript.is_generated:
                try:
                    fetched_transcript = transcript.fetch()
                    full_text = " ".join([entry['text'] for entry in fetched_transcript])
                    return full_text, transcript.language_code
                except Exception:
                    continue

        st.warning(f"No transcripts available for video {video_id}.")
        return "", ""

    except TranscriptsDisabled:
        st.warning(f"Transcripts are disabled for video {video_id}.")
        return "", ""
    except NoTranscriptAvailable:
        st.warning(f"No transcripts available for video {video_id}.")
        return "", ""
    except Exception as e:
        st.error(f"Error fetching transcript for video {video_id}: {e}")
        return "", ""

def build_style_instructions():
    """
    Based on the user's checkbox selections, build a string that will be passed to ChatGPT
    to guide the style of the summary.
    """
    instructions = []
    if st.session_state.get("step_by_step"):
        instructions.append("Write the summary step-by-step.")
    if st.session_state.get("story_based"):
        instructions.append("Use a story-based approach.")
    if st.session_state.get("with_examples"):
        instructions.append("Provide relevant examples.")
    if st.session_state.get("bullet_points"):
        instructions.append("Use bullet points.")
    if st.session_state.get("highlight_points"):
        instructions.append("Emphasize the major/highlight points.")

    # Join them into one final instruction string
    if instructions:
        return " ".join(instructions)
    else:
        return ""

def generate_summary(text: str, summary_type: str, api_url: str, api_key: str, language_code: str, style_text: str) -> str:
    """
    Generate a short or detailed summary using ChatGPT, in the same language as the transcript,
    with additional style instructions if provided.
    """
    # Base prompt depends on summary_type
    if summary_type == 'short':
        prompt = (
            f"Summarize the following YouTube video transcript in {language_code} in 3 sentences:\n\n{text}"
        )
        max_tokens = 150
    elif summary_type == 'detailed':
        prompt = (
            f"Provide a detailed summary of the following YouTube video transcript in {language_code}:\n\n{text}"
        )
        max_tokens = 300
    else:
        return ""

    # If style instructions exist, append them
    if style_text:
        prompt += f"\n\nAdditionally, please follow these style guidelines: {style_text}"

    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are an intelligent system that summarizes text based on specified style."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": max_tokens,
        "n": 1
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        return summary
    else:
        st.error(f"Error generating summary: {response.text}")
        return ""

def text_to_audio(text: str, language_code: str) -> BytesIO:
    """
    Convert text to audio using gTTS and return as a BytesIO object.
    """
    try:
        # Extract the primary language (e.g., 'en' from 'en-US')
        primary_language = language_code.split('-')[0] if '-' in language_code else language_code

        # Check if language is supported by gTTS
        supported_languages = tts_langs()
        if primary_language not in supported_languages:
            st.warning(f"Language '{language_code}' not supported for audio playback. Defaulting to English.")
            primary_language = 'en'

        tts = gTTS(text=text, lang=primary_language)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error converting text to audio: {e}")
        return BytesIO()


# -----------------------------------------------------------------
#  4. CORE LOGIC FUNCTIONS
# -----------------------------------------------------------------
def process_channels(channel_input: str, style_text: str):
    """
    1) Resolve each channel URL -> channel_id.
    2) Fetch channel details + topics (once).
    3) Fetch the latest 3 videos from the past 7 days (once).
    4) Generate short summaries for each of these videos with style instructions and save to videos.csv.
    5) Return channel_ids processed during this run.
    """
    global channels_df, videos_df

    processed_ids = []
    # Split by new lines, strip whitespace
    channel_urls = [line.strip() for line in channel_input.strip().split('\n') if line.strip()]

    for url in channel_urls:
        if not url:
            continue

        channel_id = extract_channel_id(url, YOUTUBE_API_KEY)
        if not channel_id:
            st.warning(f"Could not resolve channel ID for URL: {url}")
            continue

        # If channel already exists, skip re-adding
        if channel_id in channels_df['channel_id'].values:
            st.info(f"Channel already in list: {url}")
            processed_ids.append(channel_id)
            continue

        # 1) Get channel description
        description = get_channel_description(YOUTUBE_API_KEY, channel_id)
        if not description:
            st.warning(f"Could not retrieve description for channel: {url}")
            continue

        # 2) Categorize channel
        topics_list = categorize_channel(description, CHATGPT_API_URL, CHATGPT_API_KEY)
        topics_str = ", ".join(topics_list)

        # 3) Fetch channel title
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.channels().list(
            part='snippet',
            id=channel_id
        )
        response = request.execute()
        items = response.get('items', [])
        if items:
            channel_title = items[0]['snippet']['title']
        else:
            channel_title = "Unknown"

        # Add new row to channels_df
        new_row = {
            'channel_id': channel_id,
            'channel_url': url,
            'channel_title': channel_title,
            'description': description,
            'topics': topics_str
        }
        channels_df.loc[len(channels_df)] = new_row
        channels_df.to_csv('data/channels.csv', index=False, sep='|')

        # 4) Fetch the latest 3 videos once, create short summaries
        latest_videos = get_latest_videos(YOUTUBE_API_KEY, channel_id, days=7)
        for vid in latest_videos:
            video_id = vid['video_id']
            video_title = vid['title']

            # If video already exists, skip
            if not videos_df[videos_df['video_id'] == video_id].empty:
                continue

            # Attempt to fetch transcript
            transcript, language_code = get_transcript(video_id)
            transcript = remove_stopwords_and_articles(transcript)

            if not transcript:
                # Skip if transcript is empty
                continue

            short_summary = generate_summary(
                text=transcript, 
                summary_type='short', 
                api_url=CHATGPT_API_URL, 
                api_key=CHATGPT_API_KEY, 
                language_code=language_code,
                style_text=style_text
            )
            # We'll leave detailed_summary empty initially
            new_video_row = {
                'video_id': video_id,
                'video_title': video_title,
                'channel_id': channel_id,
                'transcript': transcript,
                'language_code': language_code,
                'short_summary': short_summary,
                'detailed_summary': ""
            }
            videos_df.loc[len(videos_df)] = new_video_row

        videos_df.to_csv('data/videos.csv', index=False, sep='|')

        st.success(f"Processed channel: {channel_title}")
        processed_ids.append(channel_id)

    return processed_ids

def display_summaries_for_channels(channel_ids, selected_topic, style_text):
    """
    Displays video summaries from videos.csv, filtered by:
      - channel_ids processed in this session
      - the selected_topic (or "All")
    """
    global channels_df, videos_df

    # 1) Filter channels to only those in channel_ids
    filtered_channels_df = channels_df[channels_df['channel_id'].isin(channel_ids)]

    # 2) If topic != "All", further filter to channels whose 'topics' contain that topic
    if selected_topic != "All":
        # 'topics' might be "Technology, Science". We check if selected_topic is in there
        topic_matched_channels = filtered_channels_df[
            filtered_channels_df['topics'].str.contains(selected_topic, case=False, na=False)
        ]
    else:
        topic_matched_channels = filtered_channels_df

    # Now gather the final list of relevant channel_ids
    final_channel_ids = topic_matched_channels['channel_id'].unique().tolist()

    # 3) Get videos only for those channels
    relevant_videos = videos_df[videos_df['channel_id'].isin(final_channel_ids)]

    # 4) Display them grouped by channel
    for channel_id in final_channel_ids:
        channel_row = channels_df[channels_df['channel_id'] == channel_id].iloc[0]
        channel_title = channel_row['channel_title']
        topics_str = channel_row['topics']

        channel_videos = relevant_videos[relevant_videos['channel_id'] == channel_id]

        for _, video_row in channel_videos.iterrows():
            video_id = video_row['video_id']
            video_title = video_row['video_title']
            transcript = video_row['transcript']
            language_code = video_row['language_code']
            short_summary = video_row['short_summary']
            detailed_summary = video_row['detailed_summary']

            with st.expander(f"{video_title}"):
                st.write(f"**Channel**: {channel_title}")
                st.write(f"**Topics**: {topics_str}")
                st.write(f"**Short Summary**: {short_summary}")

                # Button for Detailed Summary
                if st.button(f"Show Detailed Summary for {video_title}", key=f"detailed_{video_id}"):
                    # Reload videos in case of concurrency
                    videos_df = load_videos()
                    row_check = videos_df[videos_df['video_id'] == video_id]
                    if not row_check.empty:
                        det_sum = row_check['detailed_summary'].values[0]
                        language_code = row_check['language_code'].values[0]
                        transcript = row_check['transcript'].values[0]
                    else:
                        det_sum = ""
                        language_code = ""
                        transcript = ""

                    # Check if 'det_sum' is NaN
                    if pd.isna(det_sum) and transcript and language_code:
                        # Generate fresh detailed summary with style instructions
                        det_sum = generate_summary(
                            text=transcript, 
                            summary_type='detailed', 
                            api_url=CHATGPT_API_URL, 
                            api_key=CHATGPT_API_KEY, 
                            language_code=language_code,
                            style_text=style_text
                        )
                        # Save
                        if not row_check.empty:
                            idx_to_update = row_check.index[0]
                            videos_df.at[idx_to_update, 'detailed_summary'] = det_sum
                            videos_df.to_csv('data/videos.csv', index=False, sep='|')
                        else:
                            st.error("Detailed summary could not be saved.")
                    elif not transcript or not language_code:
                        det_sum = ""

                    if det_sum:
                        st.write(f"**Detailed Summary**: {det_sum}")

                # Audio playback for short summary
                if st.button(f"Listen to Short Summary for {video_title}", key=f"audio_{video_id}"):
                    if language_code:
                        audio_file = text_to_audio(short_summary, language_code)
                        st.audio(audio_file, format='audio/mp3')
                    else:
                        audio_file = text_to_audio(short_summary, 'en')
                        st.audio(audio_file, format='audio/mp3')

                # Link to watch on YouTube
                st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id})")

# -----------------------------------------------------------------
#  5. MAIN APP
# -----------------------------------------------------------------
def main():
    st.title("YouTube Channel Summarizer")

    # Two main columns: col_main (for video summaries) and col_topics (for topic filters).
    col_main, col_topics = st.columns([4, 1])

    with st.sidebar:
        st.header("Add YouTube Channels")
        channel_input = st.text_area("Enter YouTube Channel URLs (one per line)", height=150)
        
        st.subheader("Summary Style")
        # Define 5 checkboxes for the user to select
        st.checkbox("Step-by-step", key="step_by_step")
        st.checkbox("Story-based", key="story_based")
        st.checkbox("With examples", key="with_examples")
        st.checkbox("Bullet points", key="bullet_points")
        st.checkbox("Highlight major points", key="highlight_points")
        
        if st.button("Process Channels"):
            style_text = build_style_instructions()
            new_processed_ids = process_channels(channel_input, style_text)
            if new_processed_ids:
                # If we already have some in st.session_state, update them
                if "processed_ids" in st.session_state:
                    combined = set(st.session_state["processed_ids"]).union(set(new_processed_ids))
                    st.session_state["processed_ids"] = list(combined)
                else:
                    st.session_state["processed_ids"] = new_processed_ids

                # Reset topic filter to "All" after new channels processed
                st.session_state["selected_topic"] = "All"

    # If we have no processed_ids in session_state yet, show nothing
    if "processed_ids" not in st.session_state or not st.session_state["processed_ids"]:
        st.info("No channels processed yet. Please enter channel URLs and click 'Process Channels'.")
        return

    # Collect all possible topics from just these processed channels
    current_ids = st.session_state["processed_ids"]
    relevant_channels = channels_df[channels_df['channel_id'].isin(current_ids)]

    unique_topics = set()
    for _, row in relevant_channels.iterrows():
        if row['topics']:
            for t in row['topics'].split(','):
                unique_topics.add(t.strip())

    sorted_topics = sorted(list(unique_topics))
    all_options = ["All"] + sorted_topics

    with col_topics:
        st.subheader("Topics Filter")
        if not sorted_topics:
            st.write("No topics found.")
            st.session_state["selected_topic"] = "All"
        else:
            # Use a radio button with a stable key to update immediately
            st.radio(
                "Select a topic:",
                options=all_options,
                key="selected_topic"
            )

    # Now read the selected topic from session_state
    topic_to_show = st.session_state.get("selected_topic", "All")

    # Build the style instruction string each time for consistency
    style_text = build_style_instructions()

    with col_main:
        st.header("Weekly Video Summaries")
        display_summaries_for_channels(st.session_state["processed_ids"], topic_to_show, style_text)


# -----------------------------------------------------------------
#  6. LAUNCH
# -----------------------------------------------------------------
if __name__ == "__main__":
    initialize_csv_files()  # Ensure CSVs exist
    channels_df = load_channels()
    videos_df = load_videos()
    main()
