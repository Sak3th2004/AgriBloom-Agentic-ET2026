import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { MD3LightTheme as DefaultTheme, PaperProvider } from 'react-native-paper';
import HomeScreen from './screens/HomeScreen';
import HistoryScreen from './screens/HistoryScreen';
import SettingsScreen from './screens/SettingsScreen';
import { translations } from './translations';

const Tab = createBottomTabNavigator();

// Agricultural Green Theme
const theme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    primary: '#2d7d2d',
    accent: '#4caf50',
    background: '#f5f5f5',
    surface: '#ffffff',
  },
};

export default function App() {
  const [language, setLanguage] = useState('en');
  const [theme_mode, setThemeMode] = useState('light');

  const t = translations[language] || translations['en'];

  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={{
            headerShown: true,
            tabBarActiveTintColor: '#2d7d2d',
            tabBarInactiveTintColor: '#999',
            headerStyle: { backgroundColor: '#2d7d2d' },
            headerTintColor: '#fff',
            headerTitleStyle: { fontWeight: 'bold', fontSize: 20 },
          }}
        >
          <Tab.Screen
            name="Home"
            options={{
              title: t.home_title,
              tabBarLabel: 'Detect',
              tabBarIcon: ({ color }) => <Text style={{ fontSize: 24 }}>🔍</Text>,
            }}
            children={() => (
              <HomeScreen language={language} translations={t} />
            )}
          />
          <Tab.Screen
            name="History"
            options={{
              title: t.history,
              tabBarLabel: 'History',
              tabBarIcon: ({ color }) => <Text style={{ fontSize: 24 }}>📋</Text>,
            }}
            children={() => (
              <HistoryScreen language={language} translations={t} />
            )}
          />
          <Tab.Screen
            name="Settings"
            options={{
              title: t.about,
              tabBarLabel: 'Settings',
              tabBarIcon: ({ color }) => <Text style={{ fontSize: 24 }}>⚙️</Text>,
            }}
            children={() => (
              <SettingsScreen
                language={language}
                setLanguage={setLanguage}
                translations={t}
              />
            )}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
