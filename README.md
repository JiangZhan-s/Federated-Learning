# FedAvg Federated Learning Project

## Overview
This project implements a federated learning framework using the Federated Averaging (FedAvg) algorithm. The goal is to enable collaborative learning across multiple clients while keeping the data decentralized and private.

## Purpose
The purpose of this project is to demonstrate the principles of federated learning, allowing multiple clients to train a shared model without sharing their local data. This approach enhances privacy and reduces the need for centralized data storage.

## Structure
The project is organized into the following main components:

- **config/**: Contains configuration files for the FedAvg algorithm.
  - `fedavg.yaml`: Configuration parameters for the FedAvg algorithm.

- **docs/**: Documentation related to the architecture and design of the project.
  - `architecture.md`: Describes the overall architecture and module relationships.

- **scripts/**: Contains scripts for running simulations.
  - `run_federated_simulation.py`: Script to initiate the federated learning simulation.

- **src/**: The main source code for the project.
  - `main.py`: Entry point for the application.
  - **data/**: Contains data handling modules.
    - `dataset.py`: Defines the dataset class for loading and managing data.
    - `preprocessing.py`: Functions for data cleaning and transformation.
  - **federation/**: Implements federated learning components.
    - `aggregator.py`: Aggregates model updates from clients.
    - `client.py`: Manages client-server communication.
    - `strategies.py`: Implements various federated learning strategies, including FedAvg.
  - **models/**: Defines the model architecture.
    - `model.py`: Contains the model structure and forward propagation methods.
  - **training/**: Manages the training process.
    - `trainer.py`: Implements the training logic and process management.

- **tests/**: Contains unit tests for the project.
  - `test_fedavg.py`: Unit tests for the FedAvg algorithm to ensure correctness and stability.

## Usage
To run the federated learning simulation, execute the following command:

```bash
python scripts/run_federated_simulation.py
```

Make sure to install the required dependencies listed in `requirements.txt` before running the simulation.

## Installation
To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Contribution
Contributions to this project are welcome. Please feel free to submit a pull request or open an issue for any suggestions or improvements.