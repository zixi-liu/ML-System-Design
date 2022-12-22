## Model Serving

### Model Serving Patterns

Inference
- A Model
- An Interpretor
- Input Data

Batch/Static Learning vs Online Learning

Importance Metrics to Optimize
- Latency, Cost (CPU, GPU, Caching infrastructure for faster data retrieval), Throughput (# of successful requests served per unit time)
  - Methods to reduce cost: GPU sharing, Multi-model serving...
  
Model's optimizing metric vs Gating metric
- Optimizing metric: accuracy, precision, recall
- Gating metric: latency, model size, GPU load

Workflow:
- Specify the serving infrastructure -> Increase model complexity -> Improve model predictive power -> Hit gating metrics -> Accept

Maintain Input Feature Lookup
- Pre-computed or aggregated features read in real-time
- NoSQL Databases: Caching and Feature Lookup

Web applications for Users
- Users make requests via web application -> Model wrapped as API service -> Python: Fast API/Flask/Django, Java: Spring/Apache Tomcat, Clipper (multiple modeling framework), Tensorflow Serving


