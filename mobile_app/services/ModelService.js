import * as ort from 'onnxruntime-react-native';

class ModelService {
  constructor() {
    this.session = null;
    this.modelLoaded = false;
    this.labels = [
      "apple_black_rot", "apple_cedar_apple_rust", "apple_healthy", "apple_scab",
      "blueberry_healthy", "cherry_(including_sour)_healthy", "cherry_(including_sour)_powdery_mildew",
      "grape_black_rot", "grape_esca_(black_measles)", "grape_healthy", "grape_leaf_blight_(isariopsis_leaf_spot)",
      "maize_(maize)_cercospora_leaf_spot_gray_leaf_spot", "maize_(maize)_common_rust_", "maize_(maize)_healthy", "maize_(maize)_northern_leaf_blight",
      "orange_haunglongbing_(citrus_greening)",
      "peach_bacterial_spot", "peach_healthy",
      "pepper,_bell_bacterial_spot", "pepper,_bell_healthy", "pepper_bell",
      "potato__early_blight", "potato__healthy", "potato__late_blight", "potato_early_blight", "potato_healthy", "potato_late_blight",
      "ragi_blast", "ragi_healthy", "ragi_rust",
      "raspberry_healthy",
      "rice_leaf_aug",
      "soybean_healthy",
      "squash_powdery_mildew",
      "strawberry_healthy", "strawberry_leaf_scorch",
      "sugarcane_healthy", "sugarcane_mosaic", "sugarcane_redrot", "sugarcane_rust", "sugarcane_yellow",
      "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_mold",
      "tomato_mosaic_virus", "tomato_septoria_leaf_spot", "tomato_spider_mites_two_spotted_spider_mite", "tomato_target_spot",
      "tomato_yellow_leaf_curl_virus", "tomato_yellowleaf",
      "wheat_healthy", "wheat_yellow_rust"
    ];
  }

  async loadModel(modelPath) {
    try {
      console.log('Loading ONNX model...');
      this.session = await ort.InferenceSession.create(modelPath);
      this.modelLoaded = true;
      console.log('Model loaded successfully');
      return true;
    } catch (error) {
      console.error('Error loading model:', error);
      return false;
    }
  }

  normalizeImage(imageData) {
    // Normalize RGB values to [-1, 1] range
    const normalized = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < imageData.length; i += 3) {
      normalized[i] = (imageData[i] / 255.0 - 0.5) * 2;
      normalized[i + 1] = (imageData[i + 1] / 255.0 - 0.5) * 2;
      normalized[i + 2] = (imageData[i + 2] / 255.0 - 0.5) * 2;
    }
    return normalized;
  }

  async predict(imagePath) {
    if (!this.modelLoaded || !this.session) {
      throw new Error('Model not loaded');
    }

    try {
      // Load and preprocess image
      const imageData = await this.loadImage(imagePath);
      const normalized = this.normalizeImage(imageData);

      // Create input tensor [1, 3, 224, 224]
      const input = new ort.Tensor('float32', normalized, [1, 3, 224, 224]);

      // Run inference
      const results = await this.session.run({ pixel_values: input });
      const output = results.logits.data;

      // Get top prediction
      let maxConfidence = Math.max(...output);
      let maxIndex = output.indexOf(maxConfidence);

      // Convert to softmax confidence
      const confidentce = Math.exp(maxConfidence) /
        output.reduce((sum, val) => sum + Math.exp(val), 0);

      return {
        disease: this.labels[maxIndex],
        confidence: confidence,
        allPredictions: Array.from(output).map((confidence, idx) => ({
          label: this.labels[idx],
          confidence: Math.exp(confidence) /
            output.reduce((sum, val) => sum + Math.exp(val), 0)
        }))
      };
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  }

  async loadImage(imagePath) {
    // This will be implemented based on React Native Image handling
    // For now, returning placeholder
    return new Uint8ClampedArray(3 * 224 * 224);
  }
}

export default new ModelService();
