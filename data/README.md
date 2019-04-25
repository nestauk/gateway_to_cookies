# raw

## gtr_projects.csv

Gateway to Research (https://gtr.ukri.org/) projects dataset on publicly 
funded research and innovation.

### Columns

index
project_id: Unique identifier, URL
start_year: Start year of project, int
research_topics: List of research topics, e.g. 'Probability', list[str]
research_subjects: List of research subjects, e.g. 'Mathematical sciences', list[str]
abstract_texts: Long Text describing the research project, str
funder_name: Who funded the research (Acronym), str


# processed

## gtr_tokenised.csv

### Columns

index
project_id: Unique identifier, URL
abstract_texts: Long Text describing the research project, str
funder_name: Who funded the research (Acronym), str
processed_documents: List of tokens extracted from `abstract_texts`, list[str]

## gtr_embedding.csv

### Columns

index
dim_0: 0-th dimension of document vector, float
...
dim_n: n-th dimension of document vector, float

## gtr_train.csv

Training split of gtr_embedding.csv with target added
from gtr_tokenised.csv

### Columns

index
dim_0: 0-th dimension of document vector, float
...
dim_n: n-th dimension of document vector, float
funder_name: Who funded the research (Acronym), str

## gtr_test.csv

Test split of gtr_embedding.csv with target added 
from gtr_tokenised.csv

### Columns

index
dim_0: 0-th dimension of document vector, float
...
dim_n: n-th dimension of document vector, float
funder_name: Who funded the research (Acronym), str
