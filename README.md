# 🎌 Anime Recommendation System

A content-based recommendation system that suggests similar anime using **cosine similarity** based on features like genre, rating, number of episodes, and popularity.

---

## 🚀 Project Overview

This project implements an Anime Recommendation System using machine learning techniques. It recommends anime based on similarity between features such as genre, ratings, and popularity.

The system uses:
- Content-Based Filtering
- Cosine Similarity
- Feature Engineering

---

## 📊 Dataset Description

The dataset contains the following information:

- Unique ID of each anime  
- Anime title  
- Genre  
- Type (TV, Movie, OVA, etc.)  
- Number of episodes  
- Average rating  
- Number of members (popularity)  

---

## ⚙️ Features Used

- 🎭 Genre (vectorized using CountVectorizer)  
- ⭐ Rating  
- 🎬 Number of Episodes  
- 👥 Members (popularity)  

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handled missing values
- Converted categorical data into numerical form

### 2. Feature Extraction
- Genre → Converted into vectors using CountVectorizer  
- Numerical features → Normalized using MinMaxScaler  

### 3. Model Building
- Combined all features into a single matrix  
- Calculated similarity using cosine similarity  

### 4. Recommendation System
- Recommends similar anime based on similarity scores  
- Supports threshold-based filtering  

---

## 🔍 Example

**Input:**
