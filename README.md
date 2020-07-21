# DealerClustering

1.Business objective is to identify and segment dealers on basis of skill level of forecasting, speed of payment and dealer's background information. 

2. Dealer Clustering for consumer companies with large distribution footprint such as OTC pharma, FMCG or electronics OEMs  to segment
and rank dealers based on Sales, forecast accuracy and payment parameters. There is also potential for QSRs using it for improving franchisee performance.

3. Input data for dealer segmentation comprises Dealer data, forecast data, sales data, payment data and product data

4. Dealer clustering model comprises of 8 sections starting with data import to results export. Section details are summarized below.

5. Section 1 - It comprises importing packages, importing data and filling missing values

6. Section 2 - Data manipulation is carried out in this section to calculate sales, forecast accuracy and payment speed at dealer level. Dealer matrix comprised
of combination of all data is prepared in this section. This part can also be executed in SQL and section removed from model as per user requirement

7. Section 3 - Transformation of dealer matrix to standard form is carried out in this section.

8. Section 4 - K-Means analysis is carried out in this section and clustering is displayed in the form of scatter plot using seaborne package.

9. Section 5 - Dimenstionality reduction is carried out in this section. For purpose of model, Principal components analysis is used.

10. Section 6 - Cluster analysis is carried out on data transformed using PCA 

11. Section 7 - Model selection is carried out in this section using If Loop. Module currently using inertia parameter or within cluster variance parameter for model 
selection. Other parameters can be used

12. Section 8 - Post selection of model results are exported to CSV file. It is kept separate to enable addition of other methods for data transfer to visualization softwares

13. Module has scope for enhancement is choice of clustering and number of clusters. Similarly, linkage to visualization software can also be enhanced.

14. Module can also be developed further if data is available for dealer on annual basis to include growth rate, staff employed, etc.


