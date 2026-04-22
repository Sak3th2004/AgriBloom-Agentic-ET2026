import React, { useState, useFocusEffect } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  FlatList,
  Alert,
} from 'react-native';
import {
  Card,
  Text,
  Button,
  Paragraph,
} from 'react-native-paper';
import * as SQLite from 'expo-sqlite';

export default function HistoryScreen({ language, translations }) {
  const [history, setHistory] = useState([]);
  const [db, setDb] = useState(null);

  useFocusEffect(
    React.useCallback(() => {
      loadHistory();
    }, [])
  );

  const initDB = async () => {
    try {
      const database = await SQLite.openDatabaseAsync('agribloom.db');
      await database.execAsync(`
        CREATE TABLE IF NOT EXISTS detections (
          id INTEGER PRIMARY KEY,
          disease TEXT,
          confidence REAL,
          date TEXT,
          timestamp INTEGER
        );
      `);
      setDb(database);
      return database;
    } catch (error) {
      console.error('Database error:', error);
    }
  };

  const loadHistory = async () => {
    const database = db || await initDB();
    try {
      const result = await database.getAllAsync(
        'SELECT * FROM detections ORDER BY timestamp DESC LIMIT 50'
      );
      setHistory(result);
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const clearHistory = () => {
    Alert.alert(
      'Clear History',
      'Are you sure you want to delete all detection history?',
      [
        { text: 'Cancel', onPress: () => {}, style: 'cancel' },
        {
          text: 'Delete',
          onPress: async () => {
            try {
              await db?.execAsync('DELETE FROM detections;');
              setHistory([]);
            } catch (error) {
              Alert.alert('Error', 'Failed to clear history');
            }
          },
          style: 'destructive',
        },
      ]
    );
  };

  const renderHistoryItem = ({ item }) => (
    <Card style={styles.historyCard}>
      <Card.Content>
        <View style={styles.historyRow}>
          <View style={styles.historyInfo}>
            <Text style={styles.diseaseNameHistory}>{item.disease}</Text>
            <Text style={styles.confidenceHistory}>
              Confidence: {(item.confidence * 100).toFixed(1)}%
            </Text>
            <Text style={styles.dateHistory}>{item.date}</Text>
          </View>
          <View
            style={[
              styles.confidenceBadge,
              { backgroundColor: item.confidence > 0.8 ? '#4caf50' : '#ff9800' },
            ]}
          >
            <Text style={styles.badgeText}>
              {(item.confidence * 100).toFixed(0)}%
            </Text>
          </View>
        </View>
      </Card.Content>
    </Card>
  );

  return (
    <View style={styles.container}>
      {history.length > 0 ? (
        <>
          <FlatList
            data={history}
            renderItem={renderHistoryItem}
            keyExtractor={(item) => item.id.toString()}
            scrollEnabled={false}
            contentContainerStyle={styles.listContent}
          />
          <Button
            mode="outlined"
            onPress={clearHistory}
            style={styles.clearButton}
            labelStyle={styles.clearButtonLabel}
          >
            Clear History
          </Button>
        </>
      ) : (
        <Card style={styles.emptyCard}>
          <Card.Content style={styles.emptyContent}>
            <Text style={styles.emptyText}>📋</Text>
            <Paragraph style={styles.emptyMessage}>
              No detection history yet
            </Paragraph>
            <Text style={styles.emptySubtext}>
              Analyze crops to build your detection history
            </Text>
          </Card.Content>
        </Card>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 12,
  },
  listContent: {
    paddingVertical: 8,
  },
  historyCard: {
    marginBottom: 10,
    borderRadius: 10,
    elevation: 2,
  },
  historyRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  historyInfo: {
    flex: 1,
    marginRight: 12,
  },
  diseaseNameHistory: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d7d2d',
    marginBottom: 4,
  },
  confidenceHistory: {
    fontSize: 13,
    color: '#666',
    marginBottom: 4,
  },
  dateHistory: {
    fontSize: 12,
    color: '#999',
  },
  confidenceBadge: {
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  badgeText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  clearButton: {
    marginTop: 12,
    borderColor: '#c62828',
  },
  clearButtonLabel: {
    color: '#c62828',
    fontSize: 14,
  },
  emptyCard: {
    marginTop: 40,
    padding: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyContent: {
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 48,
    marginBottom: 16,
  },
  emptyMessage: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2d7d2d',
    textAlign: 'center',
  },
  emptySubtext: {
    fontSize: 13,
    color: '#999',
    marginTop: 8,
    textAlign: 'center',
  },
});
