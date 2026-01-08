# WikiElites
This repository contains all documents related to the research on **Occupational Inheritance and Kinship**.  
The study focuses on analyzing the relationships between notable individuals, their occupations, and how these factors have evolved over time. Please see the details in our paper below:

- Sirui Lai, Liang Wang, Dingqi Yang*, "Occupational Inheritance and Kinship: An Empirical Study of Historical Social Elites on Wikidata," IEEE Transactions on Computational Social Systems (TCSS), 2026.  DOI: 10.1109/TCSS.2026.3652200


---

## 📂 Dataset Description



### 1. `Q_R_Q_extended.txt`
- **Basic Information:**  
  ```
  Person1_Wikicode | Relationship_Wikicode | Person2_Wikicode
  ```
- **Additional Information:**  
  - Date of birth, death, and occupation for each person.
  - If **birth/death dates** are missing:
    - Use the earliest estimated birth date from the paper.
    - Use the latest estimated death date from the paper.
  - **Occupations** are organized in a hierarchical structure with **Level 1** as the top-level category, followed by **Level 2** and **Level 3**, representing increasingly fine-grained classifications.

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
  * M. Laouenan, P. Bhargava, J.-B. Eym´eoud, O. Gergaud, G. Plique,
and E. Wasmer, “A cross-verified database of notable people, 3500bc-
2018ad,” Scientific Data, vol. 9, no. 1, p. 290, 2022.
---
## 📊 Visualization Tools
- **Python 3.x**
  - `pandas`, `numpy`, `matplotlib`, `seaborn` for data processing and basic visualization.
- **Tableau**
  - For advanced visualizations and interactive dashboards.

## Reference

If you use our code or data, please cite our paper:

```bibtex
@article{lai2026occupational,
  title   = {Occupational Inheritance and Kinship: An Empirical Study of Historical Social Elites on Wikidata},
  author  = {Lai, Sirui and Wang, Liang and Yang, Dingqi},
  journal = {IEEE Transactions on Computational Social Systems},
  year    = {2026},
  doi     = {10.1109/TCSS.2026.3652200}
}
