# 🎬 AgriBloom Video Demo Script & Recording Guide

## Video Demo Overview
- **Duration**: ~4 minutes
- **Format**: MP4 (1080p @ 30fps)
- **Audio**: Clear narration + background music
- **Subtitles**: English + Hindi

---

## 📹 Scene-by-Scene Script

### SCENE 1: Introduction (0:00 - 0:30)
**Visual**: App icon, phone home screen
**Narration**:
> "Meet AgriBloom - AI-powered crop disease detection for Indian farmers. Imagine having an expert agronomist in your pocket, available 24/7, in your own language, completely offline."

**Action**:
- Show AgriBloom icon
- Swipe to reveal app on home screen

---

### SCENE 2: App Launch (0:30 - 1:00)
**Visual**: App splash screen → Home screen
**Narration**:
> "Whether you're in a remote village or a city farm, AgriBloom works everywhere. No internet? No problem. Works offline, in 7 Indian languages."

**Action**:
- Launch app
- Show language selector (Hindi shown)
- Tap "detect" button

---

### SCENE 3: Photo Capture (1:00 - 1:45)
**Visual**: Take photo of tomato leaf with disease
**Narration**:
> "See a diseased leaf? Just take a photo with your phone camera. Point at the infected area, capture, and let AI do the analysis."

**Action**:
- Tap "📸 Take Photo" button
- Aim at tomato leaf with blight
- Snap photo
- Show preview

---

### SCENE 4: AI Analysis & Results (1:45 - 2:45)
**Visual**: Loading animation → Results screen
**Narration**:
> "In just 2-3 seconds, our Vision Transformer model analyzes the image across 54 disease categories with 90% accuracy. Here we've detected tomato late blight - a common disease affecting millions of Indian farmers."

**Action**:
- Show loading spinner "Analyzing crop..."
- Results appear: "Tomato Late Blight"
- Confidence: "87%"
- Show recommendation card with treatment

**Recommendation shown**:
> "Remove infected parts. Use copper fungicide. Avoid overhead watering."

---

### SCENE 5: Multi-Language Feature (2:45 - 3:15)
**Visual**: Switch language to Hindi
**Narration**:
> "The same results, instantly translated. Hindi, Tamil, Kannada, Telugu - AgriBloom speaks your language."

**Action**:
- Tap settings
- Select "हिंदी (Hindi)"
- Results re-display in Hindi
- Show text in Devanagari script

---

### SCENE 6: History & Tracking (3:15 - 3:45)
**Visual**: History screen with past detections
**Narration**:
> "Track your farm health over time. History shows every detection with confidence scores and dates. Build a record of your crop's wellness."

**Action**:
- Navigate to History tab
- Show list of past detections
- Swipe through multiple entries
- Show dates and confidence badges

---

### SCENE 7: Tech Specs (3:45 - 4:00)
**Visual**: Settings screen showing model info
**Narration**:
> "Powered by Vision Transformer - the latest AI breakthrough - with 90.64% accuracy, supporting 54 diseases across 12 crops. Completely offline, works instantly."

**Action**:
- Show Settings → Model Information
- Highlight: 90.64% accuracy, 54 classes, 328MB

---

### SCENE 8: Call to Action (4:00 - 4:15)
**Visual**: App home screen with logo
**Narration**:
> "AgriBloom. Empowering Indian farmers with AI. Download today and protect your crops."

**Visual Elements**:
- App icon
- Download QR code
- "Made for Indian Farmers" tagline
- AgriBloom logo

---

## 🎬 Recording Instructions

### Equipment Needed
- Smartphone (Android or iOS)
- MacBook/Windows laptop (for screen recording)
- Quiet environment
- Microphone (headset or built-in)

### Step 1: Setup Screen Recording
**macOS**:
```bash
# Using QuickTime
1. Open QuickTime Player
2. File → New Screen Recording
3. Click red button to start
```

**Windows**:
```bash
# Using OBS Studio (Free)
1. Download OBS Studio
2. Add Display Capture source
3. Set resolution to 1080p
4. Start recording
```

### Step 2: Prepare Phones
- Clear notifications
- Set to airplane mode except WiFi
- Language set to different languages
- Have tomato leaf photo ready

### Step 3: Record Footage
1. Record each scene separately
2. Do multiple takes for smooth transitions
3. Record at 1080p 30fps minimum
4. Ensure good lighting

### Step 4: Add Narration
**Using Audacity**:
```bash
1. Import screen recording
2. Add audio track
3. Record narration while watching video
4. Sync timing
5. Export as MP4
```

### Step 5: Add Subtitles
**Using VLC or Subtitle Edit**:
```
1. Create .srt subtitle file
2. Match narration timing
3. Embed subtitles in final video
```

### Step 6: Final Output
```bash
# Using FFmpeg
ffmpeg -i video.mov -i audio.wav -c:v libx264 -preset fast -c:a aac -b:a 128k output.mp4
```

---

## 🎵 Music & Background

### Recommended BGM
- Uplifting background music (no lyrics)
- Royalty-free: YouTube Audio Library
- Duration: Full 4 minutes
- Keep volume low (background level)

### Audio Levels
- Narration: -3dB
- Background music: -20dB
- No sound effects needed

---

## 📊 Presentation Talking Points

### 1. Problem (30s)
"India is an agrarian nation. 140 million farmers depend on crop yields. Even small diseases can devastate entire harvests. But accessing expert diagnostics is expensive and slow."

### 2. Solution (1m)
"AgriBloom brings agricultural AI to every farmer's pocket. Offline. Free. In their own language. With 90%+ accuracy detection."

### 3. Technology (45s)
"We built this using Vision Transformer - winning AI model for image recognition. Trained on 141K+ real crop disease images. Exported to ONNX for mobile optimization."

### 4. Impact (1m)
"A single detection can save ₹50,000 in crop loss. Scaled across just 1% of Indian farms, that's ₹700 Crore in saved agricultural output. And our model can scale globally."

### 5. Hackathon Advantage (30s)
"Most agricultural apps achieve 70-80% accuracy. We achieved 90.64%. Most need internet. We work offline. Most are English-only. We support 7 languages."

---

## 📸 B-Roll Ideas

While recording, capture:
1. Hands holding phone with app
2. Close-ups of diseased leaves
3. Confident taps on buttons
4. Results displaying
5. Green farms/crops background
6. Farmer faces (optional)
7. Different language text

---

## ✅ Quality Checklist

Before submitting:
- [ ] Video is 1080p minimum
- [ ] Audio is clear (no background noise)
- [ ] All 4 minutes of content recorded
- [ ] Narration is synchronized
- [ ] Transitions are smooth
- [ ] Text is readable
- [ ] Language translations show clearly
- [ ] Model accuracy numbers visible
- [ ] Final video is < 500MB
- [ ] Video plays on all devices

---

## 🚀 Final Submission

### Files to Submit
1. `agribloom_demo.mp4` - Main demo video
2. `agribloom_tech_overview.pdf` - Technical details
3. `agribloom_model_report.pdf` - Accuracy metrics
4. `README.md` - Setup instructions

### Video Filename Format
`AgriBloom_ETHackathon2026_Demo_[YourName].mp4`

### Upload Locations
- Google Drive / Dropbox (5GB free)
- YouTube (Unlisted)
- Direct submission to hackathon portal

---

## 📝 Timestamps for Judges

Jump to specific points:
- **0:00-0:30** - Intro & problem
- **0:30-1:45** - App features
- **1:45-2:45** - AI detection working
- **2:45-3:15** - Multi-language proof
- **3:15-3:45** - History & tracking
- **3:45-4:00** - Tech specs
- **4:00-4:15** - Call to action

---

Made with 💚 for ET AI Hackathon 2026
