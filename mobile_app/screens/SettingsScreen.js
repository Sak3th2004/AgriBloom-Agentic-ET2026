import React from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  Linking,
} from 'react-native';
import {
  Card,
  Text,
  Button,
  Paragraph,
  Divider,
} from 'react-native-paper';
import { RadioButton } from 'react-native-paper';

const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिंदी (Hindi)' },
  { code: 'ta', name: 'தமிழ் (Tamil)' },
  { code: 'kn', name: 'ಕನ್ನಡ (Kannada)' },
  { code: 'te', name: 'తెలుగు (Telugu)' },
  { code: 'mr', name: 'मराठी (Marathi)' },
  { code: 'gu', name: 'ગુજરાતી (Gujarati)' },
];

export default function SettingsScreen({
  language,
  setLanguage,
  translations,
}) {
  return (
    <ScrollView style={styles.container}>
      {/* Language Selection */}
      <Card style={styles.card}>
        <Card.Content>
          <Text variant="titleMedium" style={styles.cardTitle}>
            🌍 {translations.language}
          </Text>
          <Divider style={styles.divider} />

          {languages.map((lang) => (
            <RadioButton.Item
              key={lang.code}
              label={lang.name}
              value={lang.code}
              status={language === lang.code ? 'checked' : 'unchecked'}
              onPress={() => setLanguage(lang.code)}
              style={styles.radioItem}
              labelStyle={styles.radioLabel}
            />
          ))}
        </Card.Content>
      </Card>

      {/* About AgriBloom */}
      <Card style={styles.card}>
        <Card.Content>
          <Text variant="titleMedium" style={styles.cardTitle}>
            ℹ️ {translations.about}
          </Text>
          <Divider style={styles.divider} />

          <Paragraph style={styles.paragraph}>
            <Text style={styles.bold}>AgriBloom v1.0</Text>
            {'\n\n'}
            AI-powered crop disease detection for Indian farmers. Recognize plant diseases instantly using your smartphone camera.
          </Paragraph>

          <Paragraph style={styles.paragraph}>
            <Text style={styles.bold}>Key Features:</Text>
            {'\n'}• 54 disease types detected
            {'\n'}• 90%+ accuracy
            {'\n'}• Works offline
            {'\n'}• Multi-language support
            {'\n'}• Free to use
          </Paragraph>

          <Paragraph style={styles.paragraph}>
            <Text style={styles.bold}>How It Works:</Text>
            {'\n'}1. Capture leaf photo
            {'\n'}2. AI analyzes instantly
            {'\n'}3. Get diagnosis & treatment
          </Paragraph>
        </Card.Content>
      </Card>

      {/* Model Information */}
      <Card style={styles.card}>
        <Card.Content>
          <Text variant="titleMedium" style={styles.cardTitle}>
            🤖 Model Information
          </Text>
          <Divider style={styles.divider} />

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Model:</Text>
            <Text style={styles.infoValue}>Vision Transformer</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Accuracy:</Text>
            <Text style={styles.infoValue}>90.64%</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Classes:</Text>
            <Text style={styles.infoValue}>54 diseases</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Size:</Text>
            <Text style={styles.infoValue}>328MB</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Speed:</Text>
            <Text style={styles.infoValue}>~2-3 seconds/image</Text>
          </View>
        </Card.Content>
      </Card>

      {/* Supported Crops */}
      <Card style={styles.card}>
        <Card.Content}>
          <Text variant="titleMedium" style={styles.cardTitle}>
            🌾 Supported Crops
          </Text>
          <Divider style={styles.divider} />

          <Text style={styles.cropList}>
            • Apple
            {'\n'}• Tomato
            {'\n'}• Potato
            {'\n'}• Grape
            {'\n'}• Cherry
            {'\n'}• Peach
            {'\n'}• Orange
            {'\n'}• Maize
            {'\n'}• Wheat
            {'\n'}• Rice
            {'\n'}• Ragi
            {'\n'}• Sugarcane
          </Text>
        </Card.Content>
      </Card>

      {/* Credits */}
      <Card style={styles.card}>
        <Card.Content>
          <Text variant="titleMedium" style={styles.cardTitle}>
            🙏 Credits
          </Text>
          <Divider style={styles.divider} />

          <Paragraph style={styles.paragraph}>
            <Text style={styles.bold}>Developed for:</Text>
            {'\n'}ET AI Hackathon 2026
            {'\n'}Problem Statement 5: Agricultural Advisory Agents
          </Paragraph>

          <Paragraph style={styles.paragraph}>
            <Text style={styles.bold}>Dataset:</Text>
            {'\n'}PlantVillage + Kaggle Agricultural Datasets
          </Paragraph>

          <Button
            mode="outlined"
            onPress={() => Linking.openURL('https://github.com')}
            style={styles.button}
          >
            View on GitHub
          </Button>
        </Card.Content>
      </Card>

      {/* Version Info */}
      <Paragraph style={styles.footerText}>
        AgriBloom v1.0 • 2026
        {'\n'}Made with 💚 for Indian Farmers
      </Paragraph>
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
    borderRadius: 10,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d7d2d',
    marginBottom: 8,
  },
  divider: {
    marginBottom: 12,
    backgroundColor: '#ddd',
  },
  radioItem: {
    paddingLeft: 0,
    marginVertical: 4,
  },
  radioLabel: {
    fontSize: 14,
  },
  paragraph: {
    fontSize: 13,
    color: '#333',
    lineHeight: 20,
    marginBottom: 12,
  },
  bold: {
    fontWeight: 'bold',
    color: '#2d7d2d',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  infoLabel: {
    fontSize: 13,
    color: '#666',
    fontWeight: '500',
  },
  infoValue: {
    fontSize: 13,
    color: '#2d7d2d',
    fontWeight: 'bold',
  },
  cropList: {
    fontSize: 13,
    color: '#333',
    lineHeight: 22,
  },
  button: {
    marginTop: 12,
    borderColor: '#2d7d2d',
  },
  footerText: {
    textAlign: 'center',
    fontSize: 12,
    color: '#999',
    paddingVertical: 24,
  },
});
