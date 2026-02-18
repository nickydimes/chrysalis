You are an expert knowledge engineer and semantic data analyst.
Your task is to extract a set of Knowledge Graph triples (subject, predicate, object) from the provided project data.

Project Data (JSON or Markdown Summary):
{DATA_CONTENT}

Knowledge Graph Ontology:
- Node Types: Context, Event, Observation, ProtocolPhase, SimModel, SimMetric, SimParameter.
- Relationship Types: RECORDS, EXHIBITS, ALIGNS_WITH, MAPS_TO, DEFINED_BY, QUANTIFIED_BY, CORRELATES_WITH, PREDICTS.

Instructions:
1. Identify entities mentioned in the data that match the Node Types.
2. Identify relationships between these entities that match the Relationship Types.
3. Be as specific as possible.
4. Output ONLY a JSON list of triples. Each triple must be a dictionary with "subject", "predicate", "object", "subject_type", and "object_type".

Output Format:
```json
[
    {
        "subject": "Riverbend Flood",
        "subject_type": "Event",
        "predicate": "EXHIBITS",
        "object": "Structural dissolution",
        "object_type": "Observation"
    },
    ...
]
```

Triples:
