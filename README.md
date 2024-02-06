## Environment Settings
> python==3.8.5 \
> scipy==1.5.4 \
> torch==1.7.0 \
> numpy==1.19.2 \
> scikit_learn==0.24.2

GPU: GeForce RTX 3080 Ti Xeon(R) Silver 4214R

## Usage
Fisrt, go into ./code, and then you can use the following commend to run our model: 
> python main.py HMDD --gpu=0

## Data Description
dd__sim.txt: Disease similarity between two diseases is calculated based on their semantic similarity in the Mesh database.

d_fea.npz: Convert dd__sim.txt into a matrix, where rows and columns correspond to disease IDs, and matrix values are the scores from the "score" column. Store it as a sparse matrix in the COO (Coordinate) format using the SciPy library.

mm__sim.txt: Obtain miRNA sequences from the miRBase database and calculate miRNA sequence similarity using the Needleman–Wunsch algorithm.

m_fea.npz: Convert mm__sim.txt into a matrix, where rows and columns correspond to miRNA IDs, and matrix values are the scores from the "score" column. Store it as a sparse matrix in the COO format using the SciPy library.

gg__sim.txt: Raw gene data is sourced from the HumanNet-XC version of the HumanNet public database. Scores are normalized using min-max scaling.

g__fea.npz: Convert gg__sim.txt into a matrix, where rows and columns correspond to gene IDs, and matrix values are the scores from the "score" column. Store it as a sparse matrix in the COO format using the SciPy library.

mg.txt: Collect relationships between miRNAs and genes from mirTarbase V8.0 (https://mirtarbase.cuhk.edu.cn/).

md.txt: Known human miRNA-disease associations are sourced from the HMDD v3.2 database (https://www.cuilab.cn/hmdd).

dg.txt: Relationships between diseases and genes come from the DisGeNET v7.0 database (https://www.disgenet.org/).

Meta-path data:

mm.npz: Set scores greater than or equal to 0.5 in mm__association.txt to 1, representing connected miRNAs. Keep only the first two columns of connected miRNA IDs and store it as a sparse matrix in COO format using the matapath.py code.

mdm.npz: Generate a sparse matrix in COO format using md.txt with matapath.py.

mgdgm.npz: Generate a sparse matrix in COO format using md.txt and mg.txt with matapath.py.

mgm.npz: Generate a sparse matrix in COO format using mg.txt with matapath.py.

dmd.npz: Generate a sparse matrix in COO format using md.txt with matapath.py.

dgd.npz: Generate a sparse matrix in COO format using dg.txt with matapath.py.

dgmgd.npz: Generate a sparse matrix in COO format using md.txt and mg.txt with matapath.py.

Disease and miRNA neighbor data:

nei_m_g.npy: Represent the gene IDs connected to miRNA using mg.txt and store them in an array.

nei_d_g.npy: Represent the gene IDs connected to diseases using dg.txt and store them in an array.

nei_m_d.npy: Represent the disease IDs connected to miRNA using md.txt and store them in an array.

nei_d_m.npy: Represent the miRNA IDs connected to diseases using md.txt and store them in an array.

Additional data for analysis:

gene-id.csv: Encode gene names starting from 0 for data analysis convenience.

disease-id.csv: Encode disease names starting from 0 for data analysis convenience.

miRNA-id.csv: Encode miRNA names starting from 0 for data analysis convenience.

Top 5 positive samples based on meta-path count:

d-pos5: The top five ranked positive sample diseases for each disease based on the number of meta-paths.

m-pos5: The top five ranked positive sample miRNAs for each miRNA based on the number of meta-paths.