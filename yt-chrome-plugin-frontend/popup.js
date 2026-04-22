const FLASK_BASE = 'http://localhost:5000';
const MAX_COMMENTS= 100
// ── DOM refs ──────────────────────────────────────────────────────────────────
const notYoutube    = document.getElementById('not-youtube');
const actionSection = document.getElementById('action-section');
const btnAnalyze    = document.getElementById('btn-analyze');
const btnReanalyze  = document.getElementById('btn-reanalyze');
const loader        = document.getElementById('loader');
const loaderText    = document.getElementById('loader-text');
const errorBox      = document.getElementById('error-box');
const resultsDiv    = document.getElementById('results');

let currentVideoId = null;

// ── Init ──────────────────────────────────────────────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  const videoId = extractVideoId(tab?.url);
  if (!videoId) {
    notYoutube.style.display = 'block';
    return;
  }
  currentVideoId = videoId;
  actionSection.style.display = 'block';
});

btnAnalyze.addEventListener('click', () => run(currentVideoId));
btnReanalyze.addEventListener('click', () => {
  resultsDiv.style.display = 'none';
  errorBox.style.display = 'none';
  actionSection.style.display = 'block';
});

// ── Utilities ─────────────────────────────────────────────────────────────────
function extractVideoId(url) {
  if (!url) return null;
  try {
    const u = new URL(url);
    if (u.hostname.includes('youtube.com')) return u.searchParams.get('v');
    if (u.hostname === 'youtu.be') return u.pathname.slice(1);
  } catch (_) {}
  return null;
}

function showLoader(text) {
  loaderText.textContent = text;
  loader.style.display = 'block';
  errorBox.style.display = 'none';
  resultsDiv.style.display = 'none';
  actionSection.style.display = 'none';
}

function hideLoader() {
  loader.style.display = 'none';
}

function showError(msg) {
  hideLoader();
  actionSection.style.display = 'block';
  errorBox.style.display = 'block';
  errorBox.textContent = '⚠ ' + msg;
}

function pct(n, total) {
  return total === 0 ? '0%' : Math.round((n / total) * 100) + '%';
}

function pctNum(n, total) {
  return total === 0 ? 0 : Math.round((n / total) * 100);
}

// ── Main flow ─────────────────────────────────────────────────────────────────
async function run(videoId) {
  try {
    showLoader('Fetching video info...');
    const videoData = await fetchVideoInfo(videoId);

    loaderText.textContent = 'Fetching comments...';
    const comments = await fetchComments(videoId);

    if (comments.length === 0) {
      showError('No comments found for this video.');
      return;
    }

    loaderText.textContent = `Analyzing ${comments.length} comments...`;
    const predictions = await predictSentiments(comments);

    const counts = { '1': 0, '0': 0, '-1': 0 };
    predictions.forEach(p => {
      const key = String(p.sentiment);
      if (key in counts) counts[key]++;
    });

    loaderText.textContent = 'Generating chart...';
    const chartUrl = await generateChart(counts);

    hideLoader();
    renderResults(videoData, predictions.length, counts, chartUrl);

  } catch (err) {
    showError(err.message || 'Something went wrong. Make sure the Flask API is running.');
  }
}

// ── API calls ─────────────────────────────────────────────────────────────────
async function fetchVideoInfo(videoId) {
  const res = await fetch(`${FLASK_BASE}/video-info`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ videoId }),
  });
  if (!res.ok) throw new Error('Flask /video-info error (' + res.status + ')');
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function fetchUrlResponse(nextPageToken,videoId,MAX_COMMENTS){
  const res = await fetch(`${FLASK_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nextPageToken,videoId,MAX_COMMENTS }),
  });
  if (!res.ok) throw new Error('Flask /search error (' + res.status + ')');
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function fetchComments(videoId) {
  let nextPageToken  = null
  let all_comments = []
  while (true){
    const res = await fetchUrlResponse(nextPageToken,videoId,MAX_COMMENTS);

    nextPageToken  = res.nextPageToken;
    (res.items || []).forEach(item =>{
      all_comments.push(item.snippet.topLevelComment.snippet.textOriginal);
    });

    if (!nextPageToken){
      break;
    }
  }
  return all_comments;
}

async function predictSentiments(comments) {
  const res = await fetch(`${FLASK_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments }),
  });
  if (!res.ok) throw new Error('Flask /predict error (' + res.status + ')');
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function generateChart(sentimentCounts) {
  const res = await fetch(`${FLASK_BASE}/generate_chart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentiment_counts: sentimentCounts }),
  });
  if (!res.ok) throw new Error('Flask /generate_chart error (' + res.status + ')');
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ── Render ────────────────────────────────────────────────────────────────────
function renderResults(videoData, total, counts, chartUrl) {
  const { snippet, statistics } = videoData;

  document.getElementById('video-title').textContent = snippet.title;
  document.getElementById('video-meta').textContent =
    `${Number(statistics?.viewCount || 0).toLocaleString()} views · ${total} comments analyzed`;

  const pos = counts['1'];
  const neu = counts['0'];
  const neg = counts['-1'];

  document.getElementById('count-pos').textContent = pos;
  document.getElementById('count-neu').textContent = neu;
  document.getElementById('count-neg').textContent = neg;

  document.getElementById('pct-pos').textContent = pct(pos, total);
  document.getElementById('pct-neu').textContent = pct(neu, total);
  document.getElementById('pct-neg').textContent = pct(neg, total);

  document.getElementById('bar-pct-pos').textContent = pct(pos, total);
  document.getElementById('bar-pct-neu').textContent = pct(neu, total);
  document.getElementById('bar-pct-neg').textContent = pct(neg, total);

  document.getElementById('bar-pos').style.width = pctNum(pos, total) + '%';
  document.getElementById('bar-neu').style.width = pctNum(neu, total) + '%';
  document.getElementById('bar-neg').style.width = pctNum(neg, total) + '%';

  document.getElementById('chart-img').src = chartUrl;

  resultsDiv.style.display = 'block';
}
