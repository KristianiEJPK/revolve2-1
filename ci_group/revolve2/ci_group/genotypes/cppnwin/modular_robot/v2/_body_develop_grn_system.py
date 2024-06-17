from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from pyrr import Vector3, Quaternion
from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2, CoreV2
import scipy.sparse as sp

@dataclass
class ModuleGRN:
    """"Goal:
        Class to hold some values for the functions in this file. 
        The class is made due to the differences between the branches.
    ----------------------------------------------------------------------
    Input:
        Module: The module as in the main branch.
        _id: The id of the module.
        _absolute_rotation: The absolute rotation of the module.
        substrate_coordinates: The coordinates of the substrate.
        turtle_direction: The direction of the front of the module.
        cell: The cell module.
        children: The children of the module.
        _parent: The parent of the module.
        direction_from_parent: The attachment face.
        """
    module: Module
    _id: int
    _absolute_rotation: int
    substrate_coordinates: tuple
    turtle_direction: int
    cell: object
    children: list
    _parent: object
    direction_from_parent: int
    forward: dict[Vector3[np.int_]]
    up: Vector3[np.int_]
    attachment_points: dict


class DevelopGRN():
    """Goal:
        Class to develop a GRN.
    ----------------------------------------------------------------------	
    """
    def __init__(self, max_modules, mode_core_mult, genotype):
        # Initialize
        self.store_gradients = False # Store gradients or not --> only for analysis, makes it slow
        self.store_location = [] # Store location of the modules --> only for analysis, makes it slow
        self.max_modules = max_modules # Maximum number of modules
        self.genotype = genotype # Genotype

        # Internal variables
        self.queried_substrate = {} # Dictionary to store the queried substrates
        self.cells, self.promotors = [], [] # List to store the cells, List to store the promotors
        self.quantity_modules = 0

        # Genotype: promotor followed by 6 values, then a promotor, etc.
        self.types_nucleotypes = 6 # Number of types of nucleotypes
        self.regulatory_tfs = 2
        self.structural_trs = len(['brick', 'joint', 'rotation'])
        self.diffusion_sites_qt = {CoreV2: 4, ActiveHingeV2: 4, BrickV2: 4} # Number of diffusion sites (probably front, back, left, right)?

        self.regulatory_transcription_factor_idx = 0 # Index of the regulatory transcription factor label
        self.regulatory_min_idx = 1 # Index of the minimum regulatory value to which gene is responsive
        self.regulatory_max_idx = 2 # Index of the maximum regulatory value to which gene is responsive
        self.transcription_factor_idx = 3 # Index of the transcription factor label
        self.transcription_factor_amount_idx = 4 # Index of the transcription factor amount upon expression
        self.diffusion_site_idx = 5 # Index of release site of the transcription factor

        # Thresholds
        self.promoter_threshold = 0.8 # Promoter threshold
        self.concentration_threshold = self.genotype[0] # Concentration threshold

        # Decay Factors
        self.concentration_decay = self.genotype[1] * 0.5 + 0.5 # Concentration decay
        self.decay_factor = {}
        self.decay_factor[CoreV2] = self.concentration_decay
        self.decay_factor[ActiveHingeV2] = self.concentration_decay
        self.decay_factor[BrickV2] = self.concentration_decay

        # Diffusion rates --> There is a possibility to make it different for each module type
        self.intra_diffusion_rate = self.genotype[2] * 0.99 + 0.01 # Intra diffusion rate
        self.intra_diffusion_rate2 = {}
        self.intra_diffusion_rate2[CoreV2] = [self.intra_diffusion_rate] * self.diffusion_sites_qt[CoreV2]
        self.intra_diffusion_rate2[ActiveHingeV2] = [self.intra_diffusion_rate] * self.diffusion_sites_qt[ActiveHingeV2]
        self.intra_diffusion_rate2[BrickV2] = [self.intra_diffusion_rate] * self.diffusion_sites_qt[BrickV2]
        self.inter_diffusion_rate = self.genotype[3] * 0.99 + 0.01 # Inter diffusion rate
        self.inter_diffusion_rate2 = {}
        self.inter_diffusion_rate2[CoreV2] = [self.inter_diffusion_rate] * self.diffusion_sites_qt[CoreV2]
        self.inter_diffusion_rate2[ActiveHingeV2] = [self.inter_diffusion_rate] * self.diffusion_sites_qt[ActiveHingeV2]
        self.inter_diffusion_rate2[BrickV2] = [self.inter_diffusion_rate] * self.diffusion_sites_qt[BrickV2]

        # Number of development steps, time step, increase scaling
        self.dev_steps = 100
        self.dt = 1
        self.increase_scaling = 100

        # Capacity of the modules
        self.capacity = {CoreV2: self.genotype[4] * 9 + 1, 
                         ActiveHingeV2: self.genotype[5] * 9 + 1, 
                         BrickV2: self.genotype[6] * 9 + 1}
        
        # Adapt genotype
        self.genotype = self.genotype[7:]

        # Get matrices
        self.create_matrices()

    def create_matrices(self):
        """Goal:
            Creates matrices required to get the concentrations of 
            the regulatory transcription (rTFs) and transcription factors (Tfs).
            ---> A, B and b, decay, x_{t}, which are part of the equation:
                ##
                Ax_{t+1} = decay * (Bx_{t} + b)
                ##
                where x represents the concentrations of the transcription factors and t
                indicates the timestep.
           
           !!! Note: 
                The matrices are created seperatedly for each transcription factor because the 
                matrix size increases quadratically with the number of nodes. 
            !!!
        -----------------------------------------------------------------------------------------------
        Input:
            diffusion_sites_qt: The number of diffusion sites for each module.
            max_modules: The maximal number of modules.
            structural_trs: The number of structural transcription factors.
            regulatory_tfs: The number of regulatory transcription factors.
            capacity: The capacity of the modules.
            dt: The time step.
        -----------------------------------------------------------------------------------------------
        Output:
            The matrices: {TF: [A, B, b, decay, x]}.
                """
        # --- Initialize
        # Get maximal possible number of nodes
        max_diffusion_sites_qt = max(self.diffusion_sites_qt.values())
        nnodes = max_diffusion_sites_qt * self.max_modules

        # Get indices of diagonal
        diagonal = np.arange(0, nnodes, dtype = np.int32)
        
        # Matrices
        self.matrices = {}

        # ---- Create matrices
        # For all possible regulatory transcription factors and transcription factors
        for tf in range(0, self.structural_trs + self.regulatory_tfs):
            # A: Future terms t + dt
            A = np.zeros((nnodes, nnodes))
            A[diagonal, diagonal] = self.capacity[ActiveHingeV2] / self.dt # Just a random value --> one of the modules values to save time
            A = sp.lil_matrix(A)

            # B: Past terms t
            B = np.zeros((nnodes, nnodes))
            B[diagonal, diagonal] = self.capacity[ActiveHingeV2] / self.dt # Just a random value --> one of the modules values to save time
            B = sp.lil_matrix(B)

            # b: Production
            b = sp.lil_matrix(np.zeros((nnodes, 1)))
            # Decay Multiplier
            decay = sp.lil_matrix(np.zeros((nnodes, 1)))
            # Concentrations
            x = sp.lil_matrix(np.zeros((nnodes, 1)))

            # --- Add to matrices
            self.matrices["TF" + str(tf + 1)] = [A, B, b, decay, x]
            
    def add2concentrations(self, indices, TF, amount):
        """Goal:
            Adds directly to the concentrations of the transcription factors. Can be used
            to set the initial concentrations.
        -----------------------------------------------------------------------------------------------
        Input:
            indices: The matrix indices.
            TF: The Transcription Factor.
            amount: The amount of the Transcription Factor to add.
        -----------------------------------------------------------------------------------------------
        Output:
            The updated matrix."""
        if self.store_gradients == False:
            for index in indices:
                if self.matrices[TF][4][index, 0] != 0:
                    self.matrices[TF][4][index, 0] += amount
                else:
                    self.matrices[TF][4][index, 0] = amount
        elif self.store_gradients == True:
            for index in indices:
                if self.matrices[TF][4][index, -1] != 0:
                    self.matrices[TF][4][index, -1] += amount
                else:
                    self.matrices[TF][4][index, -1] = amount
        
    def set_diagonal(self, diagonal, type_new):
        """Goal:
            Sets the diagonal of the matrix.
        -----------------------------------------------------------------------------------------------
        Input:
            diagonal: The matrix indices.
            type_new: The new type of the module.
        -----------------------------------------------------------------------------------------------
        Output:
            The updated matrix."""
        if type_new != ActiveHingeV2: # Is default value already
            for TF in self.matrices.keys():
                self.matrices[TF][0][diagonal, diagonal] = self.capacity[type_new] / self.dt
                self.matrices[TF][1][diagonal, diagonal] = self.capacity[type_new] / self.dt
        
    def set_intradiffusion(self, new_cell):
        """Goal:
            Sets the diffusion between sides of the same cell. Assumed that exchange takes place between
            neighbouring sides.
        -----------------------------------------------------------------------------------------------
        Input:
            new_cell: The new cell."""
        # Intitialize
        diffusion_constants = self.intra_diffusion_rate2[type(new_cell.developed_module.module)]
        # Fill 
        for TF in self.matrices.keys():
            for ds in range(0, len(new_cell.indices)):
                # --- Get ds next
                dsnext = ds + 1 if ds + 1 < len(new_cell.indices) else 0
                # --- Average diffusion constant
                avg_diffusion = 0.5 * (diffusion_constants[ds] + diffusion_constants[dsnext])
                # --- Forward diffusion
                # A
                self.matrices[TF][0][new_cell.indices[ds], new_cell.indices[ds]] += 0.5 * avg_diffusion
                if self.matrices[TF][0][new_cell.indices[ds], new_cell.indices[dsnext]] != 0:
                    self.matrices[TF][0][new_cell.indices[ds], new_cell.indices[dsnext]] -= 0.5 * avg_diffusion
                else:
                    self.matrices[TF][0][new_cell.indices[ds], new_cell.indices[dsnext]] = -0.5 * avg_diffusion
                # B
                self.matrices[TF][1][new_cell.indices[ds], new_cell.indices[ds]] -= 0.5 * avg_diffusion
                if self.matrices[TF][1][new_cell.indices[ds], new_cell.indices[dsnext]] != 0:
                    self.matrices[TF][1][new_cell.indices[ds], new_cell.indices[dsnext]] += 0.5 * avg_diffusion
                else:
                    self.matrices[TF][1][new_cell.indices[ds], new_cell.indices[dsnext]] = 0.5 * avg_diffusion
                # --- For next one back
                # A
                self.matrices[TF][0][new_cell.indices[dsnext], new_cell.indices[dsnext]] += 0.5 * avg_diffusion
                if self.matrices[TF][0][new_cell.indices[dsnext], new_cell.indices[ds]] != 0:
                    self.matrices[TF][0][new_cell.indices[dsnext], new_cell.indices[ds]] -= 0.5 * avg_diffusion
                else:
                    self.matrices[TF][0][new_cell.indices[dsnext], new_cell.indices[ds]] = -0.5 * avg_diffusion
                # B
                self.matrices[TF][1][new_cell.indices[dsnext], new_cell.indices[dsnext]] -= 0.5 * avg_diffusion
                if self.matrices[TF][1][new_cell.indices[dsnext], new_cell.indices[ds]] != 0:
                    self.matrices[TF][1][new_cell.indices[dsnext], new_cell.indices[ds]] += 0.5 * avg_diffusion
                else:
                    self.matrices[TF][1][new_cell.indices[dsnext], new_cell.indices[ds]] = 0.5 * avg_diffusion

    def set_interdiffusion(self, source_cell, new_cell, slot):
        """Goal:
            Sets the diffusion between cells. Assumed that always attaced to back of the new cell.
        -----------------------------------------------------------------------------------------------
        Input:
            source_cell: The source cell.
            new_cell: The new cell.
            slot: The slot of the parent cell.
        -----------------------------------------------------------------------------------------------
        Output:
            The updated matrix."""

        # Initialize
        diffusion_constant1 = self.inter_diffusion_rate2[type(source_cell.developed_module.module)][slot]
        diffusion_constant2 = self.inter_diffusion_rate2[type(new_cell.developed_module.module)][CoreV2.BACK]

        # Average diffusion constant
        avg_diffusion = 0.5 * (diffusion_constant1 + diffusion_constant2)

        # Fill
        for TF in self.matrices.keys():
            # Forward diffusion
            # A
            self.matrices[TF][0][source_cell.indices[slot], source_cell.indices[slot]] += 0.5 * avg_diffusion
            if self.matrices[TF][0][source_cell.indices[slot], new_cell.indices[CoreV2.BACK]] != 0:
                self.matrices[TF][0][source_cell.indices[slot], new_cell.indices[CoreV2.BACK]] -= 0.5 * avg_diffusion
            else:
                self.matrices[TF][0][source_cell.indices[slot], new_cell.indices[CoreV2.BACK]] = -0.5 * avg_diffusion
            
            # B
            self.matrices[TF][1][source_cell.indices[slot], source_cell.indices[slot]] -= 0.5 * avg_diffusion
            if self.matrices[TF][1][source_cell.indices[slot], new_cell.indices[CoreV2.BACK]] != 0:
                self.matrices[TF][1][source_cell.indices[slot], new_cell.indices[CoreV2.BACK]] += 0.5 * avg_diffusion
            else:
                self.matrices[TF][1][source_cell.indices[slot], new_cell.indices[CoreV2.BACK]] = 0.5 * avg_diffusion
            # For next one back
            # A
            self.matrices[TF][0][new_cell.indices[CoreV2.BACK], new_cell.indices[CoreV2.BACK]] += 0.5 * avg_diffusion
            if self.matrices[TF][0][new_cell.indices[CoreV2.BACK], source_cell.indices[slot]] != 0:
                self.matrices[TF][0][new_cell.indices[CoreV2.BACK], source_cell.indices[slot]] -= 0.5 * avg_diffusion
            else:
                self.matrices[TF][0][new_cell.indices[CoreV2.BACK], source_cell.indices[slot]] = -0.5 * avg_diffusion
            # B
            self.matrices[TF][1][new_cell.indices[CoreV2.BACK], new_cell.indices[CoreV2.BACK]] -= 0.5 * avg_diffusion
            if self.matrices[TF][1][new_cell.indices[CoreV2.BACK], source_cell.indices[slot]] != 0:
                self.matrices[TF][1][new_cell.indices[CoreV2.BACK], source_cell.indices[slot]] += 0.5 * avg_diffusion
            else:
                self.matrices[TF][1][new_cell.indices[CoreV2.BACK], source_cell.indices[slot]] = 0.5 * avg_diffusion


    def set_production(self, index, TF, amount):
        """Goal:
            Sets the production of the transcription factors.
        -----------------------------------------------------------------------------------------------
        Input:
            index: The matrix index.
            TF: The Transcription Factor.
            amount: The amount of the Transcription Factor to add.
        -----------------------------------------------------------------------------------------------
        Output:
            The updated matrix."""
        if self.matrices[TF][2][index, 0] != 0:
            self.matrices[TF][2][index, 0] += amount
        else:
            self.matrices[TF][2][index, 0] = amount
    
    def set_decay(self, new_cell):
        """Goal:
            Sets the decay of the transcription factors.
        -----------------------------------------------------------------------------------------------
        Input:
            new_cell: The new cell.
        -----------------------------------------------------------------------------------------------
        Output:
            The updated matrix."""
        for TF in self.matrices.keys():
            for idx in new_cell.indices:
                if self.matrices[TF][3][idx] != 0:
                    self.matrices[TF][3][idx] += self.decay_factor[type(new_cell.developed_module.module)]
                else:
                    self.matrices[TF][3][idx] = self.decay_factor[type(new_cell.developed_module.module)]
            
    def get_concentrations(self):
        """Goal:
            Get the concentrations at t + 1 of the transcription factors."""
        for TF in self.matrices.keys():
            if self.store_gradients == False:
                # B * x_{i}
                v = np.dot(self.matrices[TF][1], self.matrices[TF][4])
                # A, decay * (Bx_{i} + b)
                self.matrices[TF][4] = sp.linalg.spsolve(self.matrices[TF][0], 
                                                self.matrices[TF][3].multiply(v + self.matrices[TF][2]))
                self.matrices[TF][4] = sp.lil_matrix(self.matrices[TF][4].reshape(-1, 1))
            elif self.store_gradients == True:
                # B * x_{i}
                v = np.dot(self.matrices[TF][1], self.matrices[TF][4][:, -1])
                # A, decay * (Bx_{i} + b)
                newc = sp.linalg.spsolve(self.matrices[TF][0], 
                                                self.matrices[TF][3].multiply(v + self.matrices[TF][2]))
                self.matrices[TF][4] = np.append(self.matrices[TF][4].toarray(), newc.reshape(-1, 1), axis = 1)
                self.matrices[TF][4] = sp.lil_matrix(self.matrices[TF][4])

    def develop(self) -> BodyV2:
        """Goal:
            Develops the body of the robot."""
        self = self.develop_body()
        # Store concentrations
        if self.store_gradients == True:
            # Save locations
            print(self.store_location)
            np.savetxt('locations.csv', self.store_location, delimiter = ',')
            # Save concentrations
            for TF in self.matrices.keys():
                tf_concentrations = self.matrices[TF][4].toarray()
                # Write to file
                np.savetxt('concentrations_' + TF + '.csv', tf_concentrations, delimiter = ',')
            

        return self.phenotype_body

    def develop_body(self):
        """Goal:
            Develops the body of the robot."""
        # Call 'gene_parser' --> decodes genes from the genotype
        self = self.gene_parser()
        # Call 'regulate' --> actually does everything
        self = self.regulate()

        return self
    
    def gene_parser(self):
        """Goal:
            Create genes from the genotype."""
        # Initialize nucleotide index
        nucleotide_idx = 0

        # Repeat as long as index is smaller than gene length
        while nucleotide_idx < len(self.genotype):
            # If the associated value is smaller than the promoter threshold
            if self.genotype[nucleotide_idx] < self.promoter_threshold:
                # If there are nucleotypes enough to compose a gene
                if (len(self.genotype) - 1 - nucleotide_idx) >= self.types_nucleotypes:
                    # Get regulatory transcription factor(s)
                    regulatory_transcription_factor = self.genotype[nucleotide_idx + self.regulatory_transcription_factor_idx + 1] # Which regulatory tf is expressed?
                    regulatory_min = np.float64(self.genotype[nucleotide_idx + self.regulatory_min_idx + 1]) # Between those two values regulatory tf expresses gene
                    regulatory_max = np.float64(self.genotype[nucleotide_idx + self.regulatory_max_idx + 1])
                    # Get transcription factor, -amount and diffusion site
                    transcription_factor = self.genotype[nucleotide_idx + self.transcription_factor_idx + 1] # Which tf is expressed?
                    transcription_factor_amount = self.genotype[nucleotide_idx + self.transcription_factor_amount_idx + 1] # Amount of increase of the tf at the diffusion site
                    diffusion_site = self.genotype[nucleotide_idx + self.diffusion_site_idx + 1] # Where the tf is expressed
                    
                    # Converts rtfs and tfs values into labels
                    range_size = 1 / (self.structural_trs + self.regulatory_tfs)
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                    for idx in range(0, len(limits) - 1):
                        # Set label for regulatory transcription factor
                        if (regulatory_transcription_factor >= limits[idx]) and (regulatory_transcription_factor < limits[idx + 1]):
                            regulatory_transcription_factor_label = 'TF' + str(idx + 1)
                        elif regulatory_transcription_factor >= limits[idx + 1]:
                            regulatory_transcription_factor_label = 'TF' + str(len(limits))
                        # Set label for transcription factor
                        if (transcription_factor >= limits[idx]) and (transcription_factor < limits[idx + 1]):
                            transcription_factor_label = 'TF' + str(idx + 1)
                        elif transcription_factor >= limits[idx + 1]:
                            transcription_factor_label = 'TF' + str(len(limits))
            
                    # Converts diffusion sites values into labels
                    range_size = 1 / self.diffusion_sites_qt[CoreV2]
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                    for idx in range(0, len(limits) - 1):
                        if limits[idx+1] > diffusion_site >= limits[idx]:
                            diffusion_site_label = idx
                        elif diffusion_site >= limits[idx + 1]:
                            diffusion_site_label = len(limits) - 1
                    
                    # Translate gene to interpretable format
                    min_rTF = min([regulatory_min, regulatory_max])
                    max_rTF = max([regulatory_min, regulatory_max])
                    gene = [regulatory_transcription_factor_label, min_rTF, max_rTF,
                                transcription_factor_label, float(transcription_factor_amount), int(diffusion_site_label)]

                    # Append gene to promoters
                    self.promotors.append(gene)

                    # Increase nucleotide index
                    nucleotide_idx += self.types_nucleotypes
            
            # Increase nucleotide index
            nucleotide_idx += 1
        
        # Convert to numpy
        self.promotors_numpy = np.array(self.promotors)

        return self

    def regulate(self):
        """Goal:
            Regulates the development."""
        self = self.maternal_injection()
        self = self.growth()

        return self

    def maternal_injection(self):
        """Goal:
            Injects maternal tf into single cell embryo and starts development of the first cell.
            The tf injected is regulatory tf of the first gene in the genetic string.
            The amount injected is the minimum for the regulatory tf to regulate its regulated product.
            """
        # ---- Initialize
        self.developed_nodes = 0
        first_gene_idx = 0
        tf_label_idx = self.regulatory_transcription_factor_idx
        min_value_idx = self.regulatory_min_idx

        # for ipromotor, promotor in enumerate(self.promotors):
        #     if promotor[self.transcription_factor_idx] in ["TF3", "TF4"]:
        #         first_gene_idx = ipromotor
        #         break
        
        # ---- Get label of regulatory transcription factor of first gene
        mother_tf_label = self.promotors[first_gene_idx][tf_label_idx]
        # ---- Get minimum amount of regulatory tf required to express the gene
        mother_tf_injection = self.promotors[first_gene_idx][min_value_idx]

        # ---- Create first cell
        first_cell = Cell()
        # Get indices
        first_cell.indices = np.arange(0, self.diffusion_sites_qt[CoreV2], dtype = np.int32)
        # Distributes minimum injection among the diffusion sites
        cinit = mother_tf_injection / self.diffusion_sites_qt[CoreV2]
        # Set initial concentrations
        self.add2concentrations(first_cell.indices, mother_tf_label, cinit)
        first_cell.transcription_factors[mother_tf_label] = self.matrices[mother_tf_label][4][first_cell.indices].toarray().flatten()
        
        # Expresses promoters of first cell and updates transcription factors
        first_cell = self.express_promoters(first_cell, CoreV2)

        # Append first cell
        self.cells.append(first_cell)

        # ---- Develop a module
        first_cell.developed_module = self.place_head(first_cell)

        # ---- Set matrix components
        # Set diagonal for the first module
        self.set_diagonal(first_cell.indices, CoreV2)
        # Set intradiffusion and decay
        self.set_intradiffusion(first_cell)
        self.set_decay(first_cell)

        # --- Increase number of developed nodes
        self.developed_nodes += len(first_cell.indices)

        return self

    def express_promoters(self, new_cell, cell_type):
        """Goal:
            Expresses the promoters of a cell and updates the transcription factors.
        -----------------------------------------------------------------------------------------------
        Input:
            self: object
            new_cell: object
            cell_type: object"""
    
        # ---- For all promotors set the production
        for promotor in self.promotors:
            # ---- Initialize variables that are used multiple times
            rTF = promotor[self.regulatory_transcription_factor_idx] # Regulatory transcription factor
            TF = promotor[self.transcription_factor_idx] # Transcription factor
            ds = promotor[self.diffusion_site_idx] # Diffusion site
            amount = promotor[self.transcription_factor_amount_idx] # Amount of transcription factor

            # ---- Expresses a Tf if its rTF is present and within the range
            if rTF in new_cell.transcription_factors.keys():
                # Sum rTF concentrations
                summed_regulatory = sum(new_cell.transcription_factors[rTF])

                # --- Increase amount of transcription factor by setting the production
                # Check if between rTF min and rTF max --> set production at diffusion site by amount x
                if (summed_regulatory >= promotor[self.regulatory_min_idx]) and (summed_regulatory <= promotor[self.regulatory_max_idx]):
                    # To adhere to the original code, we update the deepcopied cell's transcription factors
                    if TF in new_cell.transcription_factors.keys():
                        new_cell.transcription_factors[TF][ds] += amount
                    else:
                        new_cell.transcription_factors[TF] = [0] * self.diffusion_sites_qt[cell_type]
                        new_cell.transcription_factors[TF][ds] = amount

                    # Add to concentration
                    self.add2concentrations(new_cell.indices, TF, amount)

        return new_cell
    
    def place_head(self, new_cell):
        """Goal: Places the head of the embryo."""
        # Initialize
        orientation = 0

        # Set variables
        self.phenotype_body = BodyV2() # Here you need to go to children--> idx --> children
        self.queried_substrate[(0, 0)] = self.phenotype_body.core
        for coordcore in [(-1, -1), (-1 , 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            self.queried_substrate[coordcore] = self.phenotype_body.core

        # Create new module
        forwards, faces = {}, {}
        for idx_attachment, attachment_face in self.phenotype_body.core_v2.attachment_faces.items():
            if idx_attachment in [0, 2]:
                forwards[idx_attachment] = Vector3([1, 0, 0])
            else:
                forwards[idx_attachment] = Vector3([-1, 0, 0])
            faces[idx_attachment] = attachment_face.attachment_points

        core_module = ModuleGRN(self.phenotype_body.core, self.quantity_modules, 
                            orientation, (0, 0), CoreV2.FRONT, new_cell, 
                            [None, None, None, None], None, None, 
                            forwards, Vector3([0, 0, 1]), faces)
        
        # Store location?
        if self.store_gradients == True:
            # Get directions
            for slot4analysis, coords4analysis in {2: (-1, 0), 3: (0, -1), 1: (0, 1), 0: (1, 0)}.items():
                self.store_location.append([0, 0, slot4analysis,
                                            coords4analysis[0], coords4analysis[1]])
        
        return core_module



    def set_increase(self, cell, tf):
        """Goal:
            Increases the amount of a transcription factor in a cell."""
        # Increase concentration at the diffusion sites
        tf_promotors = np.where(self.promotors_numpy[:, self.transcription_factor_idx] == tf)[0] # Where the trancription factor matches the tf
        
        for tf_promotor_idx in tf_promotors:
            ds = self.promotors[tf_promotor_idx][self.diffusion_site_idx]
            amount = self.promotors[tf_promotor_idx][self.transcription_factor_amount_idx] / self.increase_scaling
            self.set_production(cell.indices[ds], tf, amount)
    
        return cell

    def growth(self):
        """Goal:
            Grows the embryo."""
        # For all development steps
        for _ in range(0, self.dev_steps):
            # ---- Current number of cells
            ncells = len(self.cells)

            # ---- For all cells check if module needs to be placed or not
            for idxc in range(0, len(self.cells)):
                cell = self.cells[idxc]
                ## Place module
                # New concentrations are initialized as 0
                # Capacity, Production, decay and diffusion properties are set
                self.place_module(cell)

                ## Early stop?
                if self.quantity_modules >= self.max_modules:
                    break
            
            # --- Calculate concentrations
            csteps = 4 # In congruence with original method?
            for _ in range(0, csteps):
                self.get_concentrations()
            
            # ---- Get calculated concentrations
            for cell2 in self.cells: 
                # Get concentrations   
                for TF in self.matrices.keys():
                    if self.store_gradients == False:
                        tfconcentrations = self.matrices[TF][4][cell2.indices].toarray().flatten()
                    else:
                        tfconcentrations = self.matrices[TF][4][cell2.indices, -1].toarray().flatten()
                    # Set concentrations?    
                    if (TF in cell2.transcription_factors.keys()) or (tfconcentrations > 0).any():
                        cell2.transcription_factors[TF] = tfconcentrations
            
            # ---- Reset Production to 0
            for TF in self.matrices.keys():
                self.matrices[TF][2] *= 0
            
            # ---- Express promoters of new cells and set production for next loop
            for icell, cell in enumerate(self.cells):
                # Express promoters of new cell and updates transcription factors
                if icell >= ncells:
                    self.express_promoters(cell, type(cell.developed_module.module))
                # Get increase for next loop
                for TF in cell.transcription_factors:
                    self.set_increase(cell, TF)
            # ---- Early stop?
            if self.quantity_modules >= self.max_modules:
                break
            
        return self

    def place_module(self, cell):
        """Goal:
            Places a module in the embryo."""
        # ---- Initializes
        # Amount of transcription factors
        tds_qt = (self.structural_trs + self.regulatory_tfs)
        # Transcription factors
        product_tfs = []
        # Module types
        modules_types = [BrickV2, ActiveHingeV2]

        # Add product tfs (Brick, Hinge and rotation)
        for tf in range(tds_qt - len(modules_types) - 1, tds_qt):
            product_tfs.append(f'TF{tf+1}')

        # Get concentrations of those tfs
        concentration1 = sum(cell.transcription_factors[product_tfs[0]]) \
            if product_tfs[0] in cell.transcription_factors.keys() else 0  # B

        concentration2 = sum(cell.transcription_factors[product_tfs[1]]) \
            if product_tfs[1] in cell.transcription_factors.keys() else 0  # A

        concentration3 = sum(cell.transcription_factors[product_tfs[2]]) \
            if product_tfs[2] in cell.transcription_factors.keys() else 0  # rotation
      
        # Chooses tf with the highest concentration --> Brick or ActiveHinge
        product_concentrations = [concentration1, concentration2]
        idx_max = product_concentrations.index(max(product_concentrations))

        # If tf concentration above a threshold
        if product_concentrations[idx_max] > self.concentration_threshold:
            # Grows in the free diffusion site with the highest concentration
            freeslots = np.array([c is None for c in cell.developed_module.children])
            if type(cell.developed_module.module) == BrickV2:
                freeslots[CoreV2.BACK] = False #np.append(freeslots, [False]) # Brick has no back
                #freeslots[-1] = False
            elif type(cell.developed_module.module) == ActiveHingeV2:
                #freeslots[1:] = False
                freeslots[CoreV2.BACK] = False
                freeslots[CoreV2.LEFT] = False
                freeslots[CoreV2.RIGHT] = False # Joint has no back nor left or right

            # If free slots
            if any(freeslots):
                # Get indices of free slots
                true_indices = np.where(freeslots)[0]
                # Values
                values_at_true_indices = np.array(cell.transcription_factors[product_tfs[idx_max]])[true_indices]
                # Max value
                max_value_index = np.argmax(values_at_true_indices)
                # Index of max is new slot (coordinates calculation --> needs to adhere to front, back, left, right, etc.)
                position_of_max_value = true_indices[max_value_index]
                slot4coordinates = position_of_max_value
                # Adapt slot for setting of children
                if type(cell.developed_module.module) == ActiveHingeV2:
                    slot = 0
                elif type(cell.developed_module.module) == BrickV2: # Position 3 is slot 2
                    slot = slot4coordinates - (1 * (slot4coordinates > CoreV2.BACK))
                else: # CoreV2
                    slot = deepcopy(slot4coordinates)

                # Get coordinates and turtle direction
                potential_module_coord, turtle_direction, forward = self.calculate_coordinates(cell.developed_module, slot4coordinates, slot)
                if (potential_module_coord not in self.queried_substrate.keys()) and (self.quantity_modules < self.max_modules - 1):
                    module_type = modules_types[idx_max]

                    # ---- Rotates only joints and if defined by concentration
                    orientation = 1 if concentration3 > 0.5 and module_type == ActiveHingeV2 else 0
                    # Get absolute rotation
                    absolute_rotation = 0
                    if (module_type == ActiveHingeV2) and (orientation == 1):
                        if (type(cell.developed_module.module) == ActiveHingeV2) and (cell.developed_module._absolute_rotation == 1):
                            absolute_rotation = 0
                        else:
                            absolute_rotation = 1
                    else:
                        if (type(cell.developed_module.module) == ActiveHingeV2) and (cell.developed_module._absolute_rotation == 1):
                            absolute_rotation = 1
                    # Adapt orientation
                    if (module_type == BrickV2) and (type(cell.developed_module.module) == ActiveHingeV2) and (cell.developed_module._absolute_rotation == 1):
                        orientation = 1

                    # Set characteristics of new model
                    # Notes: new_module is the same as child, slot is the same as attachment index
                    angle = orientation * (math.pi / 2.0)
                    new_module = module_type(angle)
                    if type(cell.developed_module.module) not in [ActiveHingeV2, BrickV2]:
                        cell.developed_module.module.attachment_faces[slot].set_child(new_module, 4)
                    else:
                        cell.developed_module.module.set_child(new_module, slot)

                    self.queried_substrate[potential_module_coord] = new_module
                    self.quantity_modules += 1

                    # Create wrapper for new module
                    up = rotate(cell.developed_module.up, forward, Quaternion.from_eulers([angle, 0, 0]))
                    module2add = ModuleGRN(new_module, str(self.quantity_modules), absolute_rotation, 
                                           potential_module_coord, turtle_direction, cell, 
                                           [None, None, None, None], cell.developed_module,
                                            slot4coordinates, {0: forward}, up,
                                            new_module.attachment_points)

                    cell.developed_module.children[slot4coordinates] = module2add
                    self.new_cell(cell, module2add, slot4coordinates)

                    # # Add to grid
                    # if type(module2add.module) == ActiveHingeV2:
                    #     self.grid[potential_module_coord[0] + self.grid_origin[0], potential_module_coord[1] + self.grid_origin[1]] = 3
                    # elif type(module2add.module) == BrickV2:
                    #     self.grid[potential_module_coord[0] + self.grid_origin[0], potential_module_coord[1] + self.grid_origin[1]] = 4
                    # elif type(module2add.module) == CoreV2:
                    #     self.grid[potential_module_coord[0] + self.grid_origin[0], potential_module_coord[1] + self.grid_origin[1]] = 1
                    
                    # Store location?
                    if self.store_gradients == True:
                        # Get directions
                        self.store_location.append([cell.developed_module.substrate_coordinates[0],
                                                    cell.developed_module.substrate_coordinates[1], 
                                                    slot4coordinates,
                                                    potential_module_coord[0], potential_module_coord[1]])
    
    def new_cell(self, source_cell, new_module, slot):
        """Goal:
            Creates a new cell and shares the concentrations at diffusion sites."""
        # Create new cell
        new_cell = Cell()

        # Set matrix indices of new cell
        ul = self.developed_nodes + self.diffusion_sites_qt[type(new_module.module)]
        new_cell.indices = np.arange(self.developed_nodes, ul, dtype = np.int32)
        self.developed_nodes += len(new_cell.indices)
        assert self.developed_nodes % 4 == 0
        self.set_diagonal(new_cell.indices, type(new_module.module))

        # Share concentrations at diffusion sites
        for tf in source_cell.transcription_factors:
            # Initialize transcription factor
            new_cell.transcription_factors[tf] = [0, 0, 0, 0]
            
        # Append new cell
        self.cells.append(new_cell)

        # Set new module
        new_cell.developed_module = new_module
        new_module.cell = new_cell

        # Set diffusion rates
        self.set_intradiffusion(new_cell)
        self.set_interdiffusion(source_cell, new_cell, slot)
        self.set_decay(new_cell)
    
    def calculate_coordinates(self, parent, slot, slot_non_adapted):
        """Goal:
            Calculate the actual 2d direction and coordinates of new module using relative-to-parent position as reference."""
        
        # ---- Apply transformation
        if type(parent.module) == CoreV2:
            attachment_point = parent.attachment_points[slot_non_adapted][4] # Middle 
            fwrd = parent.forward[slot_non_adapted]
        else:
            attachment_point = parent.attachment_points[slot_non_adapted] # Middle
            fwrd = parent.forward[0] # Only one
        
        forward = rotate(fwrd, parent.up, attachment_point.orientation)
        parent_pos = np.array([parent.substrate_coordinates[0], parent.substrate_coordinates[1], 0])
        position = vec3_int(parent_pos + forward) 

        # ---- Get direction
        # dic = {CoreV2.FRONT: 0, CoreV2.LEFT: 1, CoreV2.BACK: 2, CoreV2.RIGHT: 3}
        # inverse_dic = {0: CoreV2.FRONT, 1: CoreV2.LEFT, 2: CoreV2.BACK, 3: CoreV2.RIGHT}
        # # Direction
        # direction = dic[parent.turtle_direction] + dic[slot]
        # if direction >= len(dic):
        #     direction = direction - len(dic)
        # turtle_direction = inverse_dic[direction]
        if forward == np.array([-1, 0, 0]):
            turtle_direction = CoreV2.BACK
        elif forward == np.array([0, -1, 0]):
            turtle_direction = CoreV2.RIGHT # Right and left seem to be switched, but otherwise it does not work!
        elif forward == np.array([1, 0, 0]):
            turtle_direction = CoreV2.FRONT
        elif forward == np.array([0, 1, 0]):
            turtle_direction = CoreV2.LEFT

        # # Get coordinates
        # if turtle_direction == CoreV2.LEFT:
        #     coordinates = (parent.substrate_coordinates[0] - 1,
        #                    parent.substrate_coordinates[1])
        # if turtle_direction == CoreV2.RIGHT:
        #     coordinates = (parent.substrate_coordinates[0] + 1,
        #                    parent.substrate_coordinates[1])
        # if turtle_direction == CoreV2.FRONT:
        #     coordinates = (parent.substrate_coordinates[0],
        #                    parent.substrate_coordinates[1] + 1)
        # if turtle_direction == CoreV2.BACK:
        #     coordinates = (parent.substrate_coordinates[0],
        #                    parent.substrate_coordinates[1] - 1)
        
        # Apply correction for 3 x 3 grid
        if (type(parent.module) == CoreV2):
            if position == np.array([-1, 0, 0]):
                position = (position[0] - 1, position[1])
            elif position == np.array([1, 0, 0]):
                position = (position[0] + 1, position[1])
            elif position == np.array([0, 1, 0]):
                position = (position[0], position[1] + 1)
            elif position == np.array([0, -1, 0]):
                position = (position[0], position[1] - 1)
        coordinates = (position[0], position[1])

        return coordinates, turtle_direction, forward

class Cell:
    """Goal:
        Class to model a cell.
    -----------------------------------------------------
    Input:
        self: object"""

    def __init__(self) -> None:
        self.developed_module = None
        self.transcription_factors = {}



def rotate(a: Vector3, b: Vector3, rotation: Quaternion) -> Vector3:
    """
    Rotates vector a a given angle around b.

    :param a: Vector a.
    :param b: Vector b.
    :param rotation: The quaternion for rotation.
    :returns: A copy of a, rotated.
    """
    cos_angle: int = int(round(np.cos(rotation.angle)))
    sin_angle: int = int(round(np.sin(rotation.angle)))

    vec: Vector3 = (
        a * cos_angle + sin_angle * b.cross(a) + (1 - cos_angle) * b.dot(a) * b
    )
    return vec


def vec3_int(vector: Vector3) -> Vector3[np.int_]:
    """
    Cast a Vector3 object to an integer only Vector3.

    :param vector: The vector.
    :return: The integer vector.
    """
    x, y, z = map(lambda v: int(round(v)), vector)
    return Vector3([x, y, z], dtype=np.int64)


    