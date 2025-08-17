Master's Thesis by Thomas Van Liefferinge at Universiteit Gent, titled:
Where did cycling accidents happen? A geospatial analysis of cycling accicents and their geographic influencers

All the Python codes were used to create graphs and perform classification analyses.
The titles of the code files should be self explanatory if you read the thesis.

Summary of what these codes where used for:
  Temporary analysis: number of cycling accidents by year, month, day of the week, hour of the day;
  Graph that shows the number of intersections with and without cycling accidents;
  Graph that shows the number of accidents with and without cyclisys;
  Creation of a balanced intersection dataset;
  Pearson correlation matrix of all variables;
  Graphs showing the relation between tram tracks and cycling accidents, and between traffic lights and cycling accidents;
  Classification analysis (all of this was done for logistic regression, random forest and XGBoost):
    Model with optimised hyperparameters
    Model removing feature groups one at a time
    Leave-One-Feature-Out (LOFO) analysis
