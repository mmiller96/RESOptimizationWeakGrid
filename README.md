This project presents an algorithmic approach to optimally allocate renewable energy resources such as Photovoltaic (PV) systems, electrolyzers, and fuel cells within the power grid of Puerto Carreño. This repository is structured to support research and development efforts aimed at improving grid stability and efficiency through renewable energy integration. The methodology and results herein are detailed in the paper "Optimal allocation of Renewable Energy Systems in a Weak Distribution Network."

Project Structure

    data/: Contains generated scenario data used by the optimization algorithm.
    network/: Includes information on the power grid of Puerto Carreño and associated weather data. Essential for creating the network and bus configuration.
    results/: Stores optimization results and figures generated from the analysis.

Key Files

    create_grid.py: Utilizes data from the network/ folder to create grid and bus configuration files.
    probs.py: Generates representative scenarios, saving them in the data/ folder.
    main.py: The primary script for running the optimization, relying on multiple supporting files for execution:
        loader.py: Defines prices, limitations, candidate nodes, etc.
        model_class.py: Contains the planning algorithm class.
        utils.py: Provides various utility functions used across the project.
    Results_*.py: Scripts for generating figures based on optimization results, mirroring figures from the original paper.

Getting Started

    Initial Setup: Ensure you have Python installed and clone this repository to your local machine.

    Customization for Your Grid:
        Modify the network and bus configuration files within the network/ folder according to your grid's specific data.
        Adjust parameters in loader.py as necessary to reflect your scenario's prices, limitations, and candidate nodes.

    Generating Scenarios:
        Run probs.py to generate new scenarios. These will be stored in the data/ folder.

    Running the Optimization:
        Execute main.py to start the optimization process. Ensure loader.py, model_class.py, and utils.py are properly configured and accessible.
        Optimized values will be output to the results/ folder as attr_*.pkl files.

Contributing

Contributions to enhance or expand the project's capabilities are welcome. Please fork the repository and submit a pull request with your proposed changes.
Citation

If you utilize this project in your research, please cite our paper:

"Optimal allocation of Renewable Energy Systems in a Weak Distribution Network."
