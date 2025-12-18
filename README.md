# WikiElites
This repository contains all documents related to the research on **Occupational Inheritance and Kinship**.  
The study focuses on analyzing the relationships between notable individuals, their occupations, and how these factors have evolved over time.

---

## 📂 Dataset Description



### 1. `Q_R_Q_extended.txt`
**Basic Information:**  
  ```
  Person1_Wikicode | Relationship_Wikicode | Person2_Wikicode
  ```
- **Additional Information:**  
  - Date of birth, death, and occupation for each person.
  - If birth/death dates are missing:
    - Use the **earliest estimated birth date** from the paper.
    - Use the **latest estimated death date** from the paper.
  - Occupation includes **Level 1**, **Level 2**, and **Level 3** categories.

---

### 2. `filtered_living_data.txt`
- **Description:**  
  Contains filtered data of living individuals, separated by nationality.  
  This enables **spatiotemporal analysis** and deeper examination of trends across countries.

---
### 3. `Q_R_Q.txt`
- **Format:**  
  ```
  Person1_Wikicode | Relationship_Wikicode | Person2_Wikicode
  ```
- **Description:**  
  Represents raw data queried from our local wiki database server.  
  Each entry indicates a relationship between two people based on Wikicode identifiers.  
  All individuals are classified as *notable people* according to the research paper:  
  *“A Cross-Verified Database of Notable People.”*

---
## 📊 Visualization Tools
- **Python 3.x**
  - `pandas`, `numpy`, `matplotlib`, `seaborn` for data processing and basic visualization.
- **Tableau**
  - For advanced visualizations and interactive dashboards.


