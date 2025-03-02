# YouTube Channel Summarizer

A **Streamlit** web application designed to help you monitor and stay up-to-date with the latest videos from your favorite YouTube channels. The app retrieves the most recent videos (within a configurable timeframe), extracts their transcripts, and generates concise summaries using ChatGPT. It can also categorize channels by topic and provide audio playback of summaries for hands-free updates.

## Key Features
- **Add YouTube Channels**: Input multiple channel URLs (supports standard channel IDs, custom URLs, user channels, or `@` handles).
- **Automated Summaries**: Fetches transcripts for the latest videos in the past 7 days (configurable) and generates short/detailed summaries.
- **Topic Filtering**: Categorizes each channel into relevant topics with the help of ChatGPT, enabling quick filtering by interest.
- **Audio Playback**: Converts short summaries into audio via Google Text-to-Speech (gTTS), supporting multiple languages.
- **Customizable Summary Styles**: Choose how the summaries are presented (step-by-step, bullet points, story-based, etc.).
- **Data Persistence**: Uses CSV files (`channels.csv` and `videos.csv`) to store channel data, video metadata, and summaries.

## Overview of Implementation
1. **Initialization**:
   - The app checks for two CSV files (`channels.csv` and `videos.csv`) in a `data` folder. If they don't exist, it creates them with the necessary headers.
2. **Channel Processing**:
   - Users enter YouTube channel URLs in the sidebar.
   - The app resolves channel IDs, fetches channel titles and descriptions, and uses ChatGPT to categorize them.
   - Channel info is saved to `channels.csv`.
3. **Video Fetching & Summaries**:
   - For each channel, the YouTube Data API retrieves the latest videos from the last 7 days (default).
   - The YouTube Transcript API attempts to fetch transcripts, which are then cleaned to reduce tokens.
   - ChatGPT generates short and (optionally) detailed summaries in the transcript’s original language.
   - Summaries, along with transcripts and language codes, are stored in `videos.csv`.
4. **Display & Filtering**:
   - The main interface shows a list of available topics derived from processed channels.
   - Users can filter by topic or view all channels.
   - Each video entry includes the short summary, a button for a more detailed summary, and an option to play the short summary audio.
   - Clicking a button generates the detailed summary on-demand and saves it to `videos.csv`.
5. **Styling Summaries**:
   - Several checkboxes let users request bullet-point, story-based, or step-by-step summaries. ChatGPT dynamically adapts to these preferences.

## Installation & Usage
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/youtube-channel-summarizer.git
    cd youtube-channel-summarizer
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Set Your API Keys**:
   - You will need valid **YouTube Data API** and **OpenAI ChatGPT** credentials.
   - You can store them in `st.secrets`, environment variables, or directly in the code (not recommended for security). 
     - **YOUTUBE_API_KEY**: A valid API key for YouTube Data API.
     - **CHATGPT_API_KEY**: A valid API key for ChatGPT usage.
4. **Run the App**:
    ```bash
    streamlit run app.py
    ```
   - The application will launch in your default web browser at `http://localhost:8501/`.

## Configuration & Customization
- **`YOUTUBE_API_KEY`**: Replace it with your own in the code or set as an environment variable/secret.
- **`CHATGPT_API_KEY`**: Likewise, replace or configure as needed.
- **`CHATGPT_API_URL`**: Points to the ChatGPT deployment endpoint; can be changed if you have a different endpoint or model.
- **CSV Filenames**: By default, `channels.csv` and `videos.csv` are used inside `data/`. You can rename or relocate them, but be sure to update references in the code.
- **Number of Videos Fetched**: Currently set to fetch the **latest 3 videos** from the past 7 days. Adjust in the `get_latest_videos(api_key, channel_id, days=7)` function if needed.
- **Token Optimization**: If you want to tweak how filler words are removed or how transcripts are truncated, you can modify the `remove_stopwords_and_articles()` function.
- **Summary Style**: By default, the app offers checkboxes for step-by-step, story-based, bullet points, etc. Add or remove preferences in the `build_style_instructions()` function.

## Contributing
Feel free to open issues or submit pull requests if you have suggestions or want to add new features (e.g., expansions to other data sources, improved GUI, or alternative TTS engines).

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software as you see fit.
