import React, { useState } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Image,
  Alert,
  Dimensions,
} from 'react-native';
import {
  Button,
  Card,
  Text,
  Activity indicator,
  Paragraph,
} from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import * as Camera from 'expo-camera';
import ModelService from '../services/ModelService';
import { diseaseRecommendations } from '../translations';

const { width } = Dimensions.get('window');

export default function HomeScreen({ language, translations }) {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      analyzeImage(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Camera access is required');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      analyzeImage(result.assets[0].uri);
    }
  };

  const analyzeImage = async (imageUri) => {
    setLoading(true);
    try {
      // Call ONNX model for inference
      const prediction = await ModelService.predict(imageUri);

      setResults({
        disease: prediction.disease,
        confidence: (prediction.confidence * 100).toFixed(2),
        recommendation: diseaseRecommendations[language]?.[prediction.disease] ||
          diseaseRecommendations['en'][prediction.disease] ||
          'Please consult a local agriculture expert.'
      });
    } catch (error) {
      Alert.alert('Error', 'Failed to analyze image. Please try again.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Upload Section */}
      <Card style={styles.card}>
        <Card.Content>
          <Text variant="headlineSmall" style={styles.sectionTitle}>
            {translations.home_subtitle}
          </Text>

          <View style={styles.buttonContainer}>
            <Button
              mode="contained"
              onPress={takePhoto}
              style={styles.button}
              contentStyle={styles.buttonContent}
              labelStyle={styles.buttonLabel}
            >
              {translations.btn_camera}
            </Button>

            <Button
              mode="outlined"
              onPress={pickImage}
              style={styles.button}
              contentStyle={styles.buttonContent}
              labelStyle={styles.buttonLabel}
            >
              {translations.btn_gallery}
            </Button>
          </View>
        </Card.Content>
      </Card>

      {/* Image Preview */}
      {image && (
        <Card style={styles.card}>
          <Card.Content>
            <Image source={{ uri: image }} style={styles.image} />
          </Card.Content>
        </Card>
      )}

      {/* Loading Indicator */}
      {loading && (
        <Card style={styles.card}>
          <Card.Content style={styles.centerContent}>
            <ActivityIndicator size="large" color="#2d7d2d" />
            <Text style={styles.loadingText}>Analyzing crop...</Text>
          </Card.Content>
        </Card>
      )}

      {/* Results Section */}
      {results && !loading && (
        <Card style={[styles.card, styles.resultCard]}>
          <Card.Content>
            <Text variant="headlineSmall" style={styles.resultTitle}>
              {translations.results}
            </Text>

            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>{translations.disease}:</Text>
              <Text style={styles.diseaseName}>{results.disease}</Text>
            </View>

            <View style={styles.resultItem}>
              <Text style={styles.resultLabel}>{translations.confidence}:</Text>
              <Text style={styles.confidenceText}>{results.confidence}%</Text>
            </View>

            <Card style={styles.recommendationCard}>
              <Card.Content>
                <Text style={styles.recommendationTitle}>
                  {translations.recommendations}:
                </Text>
                <Paragraph style={styles.recommendationText}>
                  {results.recommendation}
                </Paragraph>
              </Card.Content>
            </Card>

            <Button
              mode="contained"
              onPress={() => {
                setImage(null);
                setResults(null);
              }}
              style={styles.resetButton}
            >
              Analyze Another Image
            </Button>
          </Card.Content>
        </Card>
      )}

      {/* Info Section */}
      <Card style={styles.card}>
        <Card.Content>
          <Text variant="titleMedium" style={styles.infoTitle}>
            How to Use:
          </Text>
          <Text style={styles.infoText}>
            1. Take or upload a photo of crop leaves
            2. App analyzes using AI model
            3. Get instant disease diagnosis
            4. Receive treatment recommendations
{'\n'}
          </Text>
          <Text style={styles.infoText}>
            ✓ Works offline - No internet needed
            ✓ 54 disease types detected
            ✓ 90%+ accuracy
          </Text>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 12,
  },
  card: {
    marginBottom: 12,
    borderRadius: 12,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2d7d2d',
    marginBottom: 16,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 12,
    justifyContent: 'space-between',
  },
  button: {
    flex: 1,
    borderRadius: 8,
  },
  buttonContent: {
    paddingVertical: 8,
  },
  buttonLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 8,
    resizeMode: 'cover',
  },
  centerContent: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 24,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: '#2d7d2d',
    fontWeight: '500',
  },
  resultCard: {
    backgroundColor: '#f0f7f0',
    borderLeftWidth: 4,
    borderLeftColor: '#2d7d2d',
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2d7d2d',
    marginBottom: 16,
  },
  resultItem: {
    marginBottom: 12,
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 8,
  },
  resultLabel: {
    fontSize: 12,
    color: '#666',
    fontWeight: '500',
    marginBottom: 4,
  },
  diseaseName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#c62828',
  },
  confidenceText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2d7d2d',
  },
  recommendationCard: {
    backgroundColor: '#fff3e0',
    marginVertical: 12,
  },
  recommendationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#e65100',
    marginBottom: 8,
  },
  recommendationText: {
    fontSize: 13,
    color: '#333',
    lineHeight: 20,
  },
  resetButton: {
    marginTop: 12,
    backgroundColor: '#2d7d2d',
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d7d2d',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 13,
    color: '#555',
    lineHeight: 20,
  },
});
