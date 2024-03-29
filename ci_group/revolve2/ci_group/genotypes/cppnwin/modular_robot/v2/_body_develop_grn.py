from copy import deepcopy
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np
from pyrr import Vector3, Quaternion
from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2, CoreV2

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
        self.max_modules = max_modules # Maximum number of modules
        self.genotype = genotype # Genotype
        self.mode_core_mult = mode_core_mult # Mode core mult --> grid 3 x 3

        # Grid
        if not self.mode_core_mult:
            self.grid = np.zeros(shape=(max_modules * 2 + 1, max_modules * 2 + 1), dtype=np.uint8)
            self.grid[max_modules + 1, max_modules + 1] = 1
            self.grid_origin = (max_modules + 1, max_modules + 1)
        else:
            self.grid = np.zeros(shape=(max_modules * 2 + 4, max_modules * 2 + 4), dtype=np.uint8)
            self.grid[max_modules + 1:max_modules + 4, max_modules + 1:max_modules + 4] = 1
            self.grid_origin = (max_modules + 2, max_modules + 2)


        # Internal variables
        self.phenotype_body = None # Phenotype body
        self.queried_substrate = {} # Dictionary to store the queried substrate
        self.cells = [] # List to store the cells
        self.promotors = [] # List to store the promotors
        self.quantity_modules = 0

        # Indices --> seems like there is a promotor followed by 6 values, then a promotor, etc.
        self.regulatory_transcription_factor_idx = 0 # Index of the regulatory transcription factor --> which regulatory tf is expressed
        self.regulatory_min_idx = 1 # Index of the minimum regulatory value gene is responsive --> lower than suppressed
        self.regulatory_max_idx = 2 # Index of the maximum regulatory value gene is responsive --> higher than suppressed
        self.transcription_factor_idx = 3 # Index of the transcription factor --> which transcription factor is expressed
        self.transcription_factor_amount_idx = 4 # Index of the transcription factor amount --> amount of increase of the tf at the diffusion site
        self.diffusion_site_idx = 5 # Index of the diffusion site --> where the tf is expressed

        # Number of nucleotides, number of diffusion sites, kind of transcription factors and number of regulatory transcription factors
        self.types_nucleotypes = 6 # Number of types of nucleotypes
        self.diffusion_sites_qt = 4 # Number of diffusion sites (probably front, back, left, right)?
        self.structural_trs = len(['brick', 'joint', 'rotation'])
        self.regulatory_tfs = 2

        # Parameters
        self.promoter_threshold = 0.8 # Promoter threshold
        self.concentration_decay = 0.005 # Concentration decay
        self.concentration_threshold = self.genotype[0] # Concentration threshold
        self.increase_scaling = 100
        self.intra_diffusion_rate = self.concentration_decay / 2
        self.inter_diffusion_rate = self.intra_diffusion_rate / 8
        self.dev_steps = 100 # Number of development steps

        # Adapt genotype
        self.genotype = self.genotype[1:]

    def develop(self) -> BodyV2:
        """Goal:
            Develops the body of the robot."""
        # Initialize
        self = self.develop_body()

        # # ---- Plot
        # # Create a custom colormap with 4 colors
        # cmap = plt.cm.colors.ListedColormap(['grey', 'red', 'black', 'white', 'blue'])

        # # Create a normalized color map
        # norm = plt.cm.colors.Normalize(vmin=0, vmax=4)

        # # Create an array of colors based on the values
        # plt.imshow(self.grid, cmap = cmap, norm = norm)
        # plt.xticks(np.arange(0, self.grid.shape[0], 1))
        # plt.yticks(np.arange(0, self.grid.shape[1], 1))
        # plt.grid(True, which='both')
        # plt.show()

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
                    regulatory_min = self.genotype[nucleotide_idx + self.regulatory_min_idx + 1] # Between those two values regulatory tf expresses gene
                    regulatory_max = self.genotype[nucleotide_idx + self.regulatory_max_idx + 1]
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
                    range_size = 1 / self.diffusion_sites_qt
                    limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                    for idx in range(0, len(limits) - 1):
                        if limits[idx+1] > diffusion_site >= limits[idx]:
                            diffusion_site_label = idx
                        elif diffusion_site >= limits[idx + 1]:
                            diffusion_site_label = len(limits) - 1
                    
                    # Translate gene to interpretable format
                    gene = [regulatory_transcription_factor_label, regulatory_min, regulatory_max,
                                transcription_factor_label, transcription_factor_amount, diffusion_site_label]

                    # Append gene to promoters
                    self.promotors.append(gene)

                    # Increase nucleotide index
                    nucleotide_idx += self.types_nucleotypes
            
            # Increase nucleotide index
            nucleotide_idx += 1
        
        # Convert to numpy
        self.promotors = np.array(self.promotors)

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
        # Initialize
        first_gene_idx = 0
        tf_label_idx = 0
        min_value_idx = 1

        # Get label of regulatory transcription factor of first gene
        mother_tf_label = self.promotors[first_gene_idx][tf_label_idx]
        # Get minimum amount of regulatory tf required to express the gene
        mother_tf_injection = float(self.promotors[first_gene_idx][min_value_idx])

        # Create first cell
        first_cell = Cell()

        # Distributes minimum injection among the diffusion sites
        first_cell.transcription_factors[mother_tf_label] = \
            [mother_tf_injection / self.diffusion_sites_qt] * self.diffusion_sites_qt
        
        # Expresses promoters of first cell and updates transcription factors
        first_cell = self.express_promoters(first_cell)
        
        # Append first cell
        self.cells.append(first_cell)

        # Develop a module
        first_cell.developed_module = self.place_head(first_cell)

        return self

    def express_promoters(self, new_cell):
        """Goal:
            Expresses the promoters of a cell and updates the transcription factors.
        -----------------------------------------------------------------------------------------------
        Input:
            self: object
            new_cell: object"""
        
        for promotor in self.promotors:
            # Get regulatory min and max values
            regulatory_min_val = min(float(promotor[self.regulatory_min_idx]),
                                        float(promotor[self.regulatory_max_idx]))
            regulatory_max_val = max(float(promotor[self.regulatory_min_idx]),
                                        float(promotor[self.regulatory_max_idx]))
            
            # Expresses a tf if its regulatory tf is present and within a range
            if new_cell.transcription_factors.get(promotor[self.regulatory_transcription_factor_idx]) \
                    and (sum(new_cell.transcription_factors[promotor[self.regulatory_transcription_factor_idx]]) \
                    >= regulatory_min_val) \
                    and (sum(new_cell.transcription_factors[promotor[self.regulatory_transcription_factor_idx]]) \
                    <= regulatory_max_val):

                # Update or add transcription factor
                if new_cell.transcription_factors.get(promotor[self.transcription_factor_idx]):
                    new_cell.transcription_factors[promotor[self.transcription_factor_idx]] \
                        [int(promotor[self.diffusion_site_idx])] += float(promotor[self.transcription_factor_amount_idx])
                else:
                    new_cell.transcription_factors[promotor[self.transcription_factor_idx]] = [0] * self.diffusion_sites_qt
                    new_cell.transcription_factors[promotor[self.transcription_factor_idx]] \
                    [int(promotor[self.diffusion_site_idx])] = float(promotor[self.transcription_factor_amount_idx])
        
        return new_cell
    
    def place_head(self, new_cell):
        """Goal: Places the head of the embryo."""
        # Initialize
        orientation = 0

        # Set variables
        self.phenotype_body = BodyV2() # Here you need to go to children--> idx --> children
        self.queried_substrate[(0, 0)] = self.phenotype_body.core
        if self.mode_core_mult:
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

        return core_module

    def growth(self):
        """Goal:
            Grows the embryo."""
        # For all development steps
        for t in range(0, self.dev_steps):

            # Develops cells in order of age --> oldest first
            for idxc in range(0, len(self.cells)):
                cell = self.cells[idxc]
                # For all transcription factors
                for tf in cell.transcription_factors:
                    # Incease amount of transcription factor in the cell
                    self.increase(tf, cell)
                    # Intra diffusion
                    self.intra_diffusion(tf, cell)
                    # Inter diffusion
                    self.inter_diffusion(tf, cell)

                # Place module
                self.place_module(cell)

                # Decay transcription factors
                for tf in cell.transcription_factors:
                    self.decay(tf, cell)
        return self

    def increase(self, tf, cell):
        """Goal:
            Increases the amount of a transcription factor in a cell."""
        # Increase concentration at the diffusion sites
        tf_promotors = np.where(self.promotors[:, self.transcription_factor_idx] == tf)[0] # Where the trancription factor matches the tf
        for tf_promotor_idx in tf_promotors:
            cell.transcription_factors[tf][int(self.promotors[tf_promotor_idx][self.diffusion_site_idx])] += \
                float(self.promotors[tf_promotor_idx][self.transcription_factor_amount_idx]) \
                / float(self.increase_scaling)
        
        return cell

    def intra_diffusion(self, tf, cell):
        """Goal:
            Performs intra diffusion of a transcription factor in a cell."""
        # For each site: first right then left
        for ds in range(0, self.diffusion_sites_qt):
            # Get left and right diffusion sites
            ds_left = ds - 1 if ds - 1 >= 0 else self.diffusion_sites_qt - 1
            ds_right = ds + 1 if ds + 1 <= self.diffusion_sites_qt - 1 else 0

            # Diffuse to right
            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_right] += self.intra_diffusion_rate
            # Diffuse to left
            if cell.transcription_factors[tf][ds] >= self.intra_diffusion_rate:
                cell.transcription_factors[tf][ds] -= self.intra_diffusion_rate
                cell.transcription_factors[tf][ds_left] += self.intra_diffusion_rate
    
    def inter_diffusion(self, tf, cell):
        """Goal: Performs inter diffusion of a transcription factor between cells."""
        # For each diffusion site
        for ds in range(0, self.diffusion_sites_qt):
            # If diffusion site is the back (non-core) and the developed cell is a hinge or a brick
            if (ds == CoreV2.BACK) and \
                    (type(cell.developed_module.module) == ActiveHingeV2 or type(cell.developed_module.module) == BrickV2):
                # If transcription factor concentration is equal or greater than the inter diffusion rate
                if cell.transcription_factors[tf][CoreV2.BACK] >= self.inter_diffusion_rate:
                    cell.transcription_factors[tf][CoreV2.BACK] -= self.inter_diffusion_rate

                    # Update or add transcription factor
                    if cell.developed_module._parent.cell.transcription_factors.get(tf):
                        cell.developed_module._parent.cell.transcription_factors[tf][cell.developed_module.direction_from_parent] += self.inter_diffusion_rate
                    else:
                        cell.developed_module._parent.cell.transcription_factors[tf] = [0] * self.diffusion_sites_qt
                        cell.developed_module._parent.cell.transcription_factors[tf][cell.developed_module.direction_from_parent] += self.inter_diffusion_rate

            # If diffusion site is not the back and the developed cell is a hinge --> also share from other sides without slots
            elif (type(cell.developed_module.module) == ActiveHingeV2) and \
                    ds in [CoreV2.LEFT, CoreV2.FRONT, CoreV2.RIGHT]:
                # If the front side is not None and transcription factor concentration is equal or greater than the inter diffusion rate
                if (cell.developed_module.children[CoreV2.FRONT] is not None) \
                        and (cell.transcription_factors[tf][ds] >= self.inter_diffusion_rate):
                    cell.transcription_factors[tf][ds] -= self.inter_diffusion_rate

                    # Update or add transcription factor
                    if cell.developed_module.children[CoreV2.FRONT].cell.transcription_factors.get(tf):
                        cell.developed_module.children[CoreV2.FRONT].cell.transcription_factors[tf][CoreV2.BACK] += self.inter_diffusion_rate
                    else:
                        cell.developed_module.children[CoreV2.FRONT].cell.transcription_factors[tf] = [0] * self.diffusion_sites_qt
                        cell.developed_module.children[CoreV2.FRONT].cell.transcription_factors[tf][CoreV2.BACK] += self.inter_diffusion_rate
            else:
                # If a developed module and the transcription factor concentration is equal or greater than the inter diffusion rate
                if (cell.developed_module.children[ds] is not None) \
                    and (cell.transcription_factors[tf][ds] >= self.inter_diffusion_rate):
                    cell.transcription_factors[tf][ds] -= self.inter_diffusion_rate

                    # Update or add transcription factor
                    if cell.developed_module.children[ds].cell.transcription_factors.get(tf):
                        cell.developed_module.children[ds].cell.transcription_factors[tf][CoreV2.BACK] += self.inter_diffusion_rate
                    else:
                        cell.developed_module.children[ds].cell.transcription_factors[tf] = [0] * self.diffusion_sites_qt
                        cell.developed_module.children[ds].cell.transcription_factors[tf][CoreV2.BACK] += self.inter_diffusion_rate
    
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
            if cell.transcription_factors.get(product_tfs[0]) else 0  # B

        concentration2 = sum(cell.transcription_factors[product_tfs[1]]) \
            if cell.transcription_factors.get(product_tfs[1]) else 0  # A

        concentration3 = sum(cell.transcription_factors[product_tfs[2]]) \
            if cell.transcription_factors.get(product_tfs[2]) else 0  # rotation
      
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
                # Index of max is new slot (coordinates calculation)
                position_of_max_value = true_indices[max_value_index]
                slot4coordinates = position_of_max_value # !!!
                # Adapt slot for setting of children
                if type(cell.developed_module.module) == ActiveHingeV2:
                    slot = 0
                elif type(cell.developed_module.module) == BrickV2:
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

                    # Add to grid
                    if type(module2add.module) == ActiveHingeV2:
                        self.grid[potential_module_coord[0] + self.grid_origin[0], potential_module_coord[1] + self.grid_origin[1]] = 3
                    elif type(module2add.module) == BrickV2:
                        self.grid[potential_module_coord[0] + self.grid_origin[0], potential_module_coord[1] + self.grid_origin[1]] = 4
                    elif type(module2add.module) == CoreV2:
                        self.grid[potential_module_coord[0] + self.grid_origin[0], potential_module_coord[1] + self.grid_origin[1]] = 1


    def decay(self, tf, cell):
        """Goal:
            Decays the amount of a transcription factor in a cell."""
        # Decay at all sites
        for ds in range(0, self.diffusion_sites_qt):
            cell.transcription_factors[tf][ds] = \
                max(0, cell.transcription_factors[tf][ds] - self.concentration_decay)
    
    def new_cell(self, source_cell, new_module, slot):
        """Goal:
            Creates a new cell and shares the concentrations at diffusion sites."""
        # Create new cell
        new_cell = Cell()

        # Share concentrations at diffusion sites
        for tf in source_cell.transcription_factors:
            # Initialize transcription factor
            new_cell.transcription_factors[tf] = [0, 0, 0, 0]

            # In the case of joint, also shares concentrations of sites without slot
            if type(source_cell.developed_module.module) == ActiveHingeV2:
                sites = [CoreV2.LEFT, CoreV2.FRONT, CoreV2.RIGHT]
                for s in sites:
                    if source_cell.transcription_factors[tf][s] > 0:
                        # Get half of the concentration
                        half_concentration = source_cell.transcription_factors[tf][s] / 2
                        # Share half of the concentration
                        source_cell.transcription_factors[tf][s] = half_concentration
                        new_cell.transcription_factors[tf][CoreV2.BACK] += half_concentration
                # Divide by the number of sites
                new_cell.transcription_factors[tf][CoreV2.BACK] /= len(sites)
            else:
                if source_cell.transcription_factors[tf][slot] > 0:
                    half_concentration = source_cell.transcription_factors[tf][slot] / 2
                    source_cell.transcription_factors[tf][slot] = half_concentration
                    new_cell.transcription_factors[tf][CoreV2.BACK] = half_concentration

        # Express promoters of new cell and updates transcription factors
        self.express_promoters(new_cell)
        # Append new cell
        self.cells.append(new_cell)
        # Set new module
        new_cell.developed_module = new_module
        new_module.cell = new_cell
    
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
        if self.mode_core_mult and (type(parent.module) == CoreV2):
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


    