# messengerAnalytics
python analysis tool for Facebook messenger data

1. Download your Facebook messages data in JSON format from Facebook.
2. Install Python dependencies and download messengerAnalytics.py
after installing textblob run this in terminal: python -m textblob.download_corpora
3. Place all Facebook conversation folders in directory: ./data/messages/*
4. Run Python script from command line

## measure and plot metrics such as:
- Avg number of messages per day during friendship
- Avg number of pictures shared per day during friendship
- Ratio of messages sent by me to messages received
- Ratio of non-stopwords sent by me to non-stopwords received
- Responsiveness
- Avg polarity of messages
- Avg polarity trend over time
- Heatmap of most used words
- Most positive and most negative messages sent by each person
