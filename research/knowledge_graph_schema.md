# Chrysalis Knowledge Graph Schema

This document defines the ontology for the Chrysalis Dynamic Knowledge Graph, designed to uncover hidden relationships between qualitative ethnographic observations and quantitative physical simulations.

## 1. Node Types (Entities)

| Type | Description |
|---|---|
| `Context` | An ethnographic source, historical record, or geographic location. |
| `Event` | A specific occurrence or social transition (e.g., "Oakhaven Drought"). |
| `Observation` | A qualitative critical element observed in the data (e.g., "loss of identity"). |
| `ProtocolPhase` | One of the Eight Steps (Purification, Containment, ..., Emergence). |
| `SimModel` | A physics simulation model (Ising, Potts, Percolation, etc.). |
| `SimMetric` | A quantitative measurement from a simulation (Susceptibility, Entropy, etc.). |
| `SimParameter` | A control variable (Temperature, Connectivity). |

## 2. Relationship Types (Predicates)

| Relationship | Source Node | Target Node | Description |
|---|---|---|---|
| `RECORDS` | `Context` | `Event` | Source describes a specific event. |
| `EXHIBITS` | `Event` | `Observation` | Event shows a specific qualitative dynamic. |
| `ALIGNS_WITH` | `Observation` | `ProtocolPhase` | Observation matches a navigation step. |
| `MAPS_TO` | `ProtocolPhase` | `SimModel` | Phase transition in model mirrors protocol step. |
| `DEFINED_BY` | `SimModel` | `SimParameter` | Model behavior controlled by parameter. |
| `QUANTIFIED_BY` | `SimModel` | `SimMetric` | Model state measured by metric. |
| `CORRELATES_WITH` | `Observation` | `SimMetric` | **Hidden Relationship:** Qualitative element matches metric signature. |
| `PREDICTS` | `SimModel` | `Event` | Model results suggest future state of event. |

## 3. Data Representation
The graph will be stored as a directed graph in **NetworkX** and persisted as a **GraphML** file (`data/knowledge_graph.graphml`) for easy visualization and a set of triples in SQLite for querying.
