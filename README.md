# MLP-Export-Value-Prediction-Project

Welcome to the MLP Export Prediction Repository 🌱🌵🌿☘. This project has to do with creating a multi-layer perceptron model that is capable of predicting the export value of agricultural products of countries around the world up to three years into the future.

## Project Description

For many countries, being able to predict the exporting power of a nation in advance can be beneficial in so many aspects. Some of them includes;
- Economic Planning and Policy Making: Accurate forecasts of export value can help the country to make more informed and prepared economic plans that can be beneficial to the nation as a whole. This can help to stabilize the economy and foster industrial growth.
- Business and Investment Decision: Companies can plan their production, inventory, and marketing strategies more effectively with reliable export forecasts. It can help in managing supply chains, reducing costs, and optimizing resource allocation. Investors also use export forecasts to evaluate the economic health of a country and make informed decisions about investing in that country.
- Risk Management and Currency Fluctations: Forecasting helps in identifying potential economic downturns or booms, allowing stakeholders to reduce risks associated with volatile market conditions. By predicting export trends, businesses can hedge against currency fluctuations and manage financial risks more effectively.
- Resource Allocation and Development: Accurate export forecasts can guide infrastructure development such as ports, logistics, and transportation networks to support increased trade volumes. It also aids in identifying and nurturing key sectors that have high export potential, thus driving sectoral development and economic diversification.

## About the Data

The dataset used for this project is made up of data extracted from the [FAOSTAT database](https://www.fao.org/faostat/en/#home), which gives open access to food and agricultural data for over 245 countries and covers years from mid 1990s to present day.

The project Dataset contains 13 csv files, each covering a category of variables relevant to food and agriculture. Here are the 13 categories covered (with the corresponding FAOSTAT source for each category);

- Consumer prices indicators - [link](https://www.fao.org/faostat/en/%23ata/CP)
- Crops production indicators - [link](https://www.fao.org/faostat/en/%23data/QCL)
- Emissions - [link1](https://www.fao.org/faostat/en/%23data/GCE) and [link2](https://www.fao.org/faostat/en/%23data/GV)
- Employment - [link](https://www.fao.org/faostat/en/%23data/OEA)
- Exchange rate -  [link](https://www.fao.org/faostat/en/%23data/PE)
- Fertilizers use - [link](https://www.fao.org/faostat/en/%23data/RFB)
- Food balances indicators - [link](https://www.fao.org/faostat/en/%23data/FBS)
- Food security indicators - [link](https://www.fao.org/faostat/en/%23data/FS)
- Food trade indicators - [link](https://www.fao.org/faostat/en/%23data/TCL)
- Foreign direct investment - [link](https://www.fao.org/faostat/en/%23data/FDI)
- Land temperature change - [link](https://www.fao.org/faostat/en/#%23ata/ET)
- Land use - [link](https://www.fao.org/faostat/en/#%23ata/RL)
- Pesticides use - [link](https://www.fao.org/faostat/en/%23data/RP)

You can see [here](https://www.fao.org/faostat/en/%23definitions) for more details about the headers in the data files. The table below provides a summary of the variables included in each file.


**Consumer price indicators**
- Consumer price, food index
- Country
- Food price inflation
-  Month
- Year
  
**Crops production indicators**
- Country
- Year
- Yield for different crop products


**Emissions**
- Country
- Crops CH4 emissions
- Crops N2O emissions
- Drained soil CO2 emissions
- Drained soil N2O emissions
- Year

**Employment**
- Country
- Employment (male and female total) in agriculture, forestry, and fishing
- Mean weekly hours worked per person (no distinction between male and female) in agriculture, forestry, and fishing
- Year
  
**Exchange rate**
- Country
- Currency
- Local currency units per USD
- Months
- Year
  
**Fertilizers use**
- Agricultural use of different categories of fertilizers
- Country
- Year

**Food balances**
- Country
- Export quantity for different crop and livestock products
- Food uses for different crop and livestock products
- Import quantity for different crop and livestock products
- Losses for crop and livestock products
- Other uses for crop and livestock products
- Year

**Food security**
- Cereal import dependency ratio
- Country
- Dietary energy supply adequacy
- Per capita food production variability
- Per capita food supply variability
- Percentage of arable land equipped for irrigation
- Political stability and absence of violence/terrorism index
- Prevalence of anaemia in women of reproductive age
- Prevalence of low birthweight
- Protein energy supply
- Value of food imports in total merchandise exports
- Year
  
**Food trade**
- Country
- Export value
- Import value
- Year

**Foreign direct investment (FDI)**
- Country
- FDI inflows to agriculture, forestry, and fishing
- FDI inflows to food, beverages, and tobacco
- FDI outflows to agriculture, forestry, and fishing
- FDI outflows to food, beverages, and tobacco
- Total FDI inflows
- Total FDI outflows
- Year
  
**Land temperature change**
- Country
- Months
- Temperature change
- Standard deviation
- Year

  
**Land use**
- Area for different categories of land use
- Country
- Year
  
**Pesticides use**
- Agricultural use for each of fungicides (and bactericides), herbicides, insecticides, pesticides, rodenticides
- Country
- Use per area of cropland for each of fungicides (and bactericides), herbicides, insecticides, pesticides, rodenticides
- Use per value of agricultural production for each of fungicides (and bactericides), herbicides, insecticides, pesticides, rodenticides
- Year

## Setup
It is recommended that this project is ran on [Google colaboratory](colab.research.google.com) using a GPU because of the large computing power required to train and evaluate the neural networks model. [Google colaboratory](colab.research.google.com) also has most of the required tools and libraries needed to execute the project. Any tools that wasn't available was installed using PIP and to do the same, just uncomment and run the required code cell.

To use this repository,
- clone it to your local machine using the code below;
  ```git clone https://github.com/Dee-ui/MLP-Export-Value-Prediction-Project.git```
- Navigate to project directory;
    ```cd MLP-Export-Value-Prediction-Project```
- Run the .ipynb Notebook using [Google colaboratory](colab.research.google.com) to explore and execute the project ;
    ```jupyter notebook MLP_project.ipynb```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please review our contribution guidelines before getting started.

## License
This project is licensed under the MIT License.

## Contacts and Acknowledgments
`Author/programmer` :- Agbonoga Dauda [email](daudaagbonoga@gmail.com)
