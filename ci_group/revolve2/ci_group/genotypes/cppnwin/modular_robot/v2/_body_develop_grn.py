import numpy as np
import random

# todo: first work out all functions to get an understanding, then change it to a class based
# nature.

def develop_body(promotors, genotype, promoter_threshold, types_nucleotypes, regulatory_transcription_factor_idx,
                regulatory_min_idx, regulatory_max_idx, transcription_factor_idx, transcription_factor_amount_idx,
                diffusion_site_idx, structural_trs, regulatory_tfs, diffusion_sites_qt,
                cells):
    # Call 'gene_parser'
    promotors = gene_parser(promotors, genotype, promoter_threshold, types_nucleotypes, regulatory_transcription_factor_idx,
                regulatory_min_idx, regulatory_max_idx, transcription_factor_idx, transcription_factor_amount_idx,
                diffusion_site_idx, structural_trs, regulatory_tfs, diffusion_sites_qt)
    # Call 'regulate'
    regulate(promotors, diffusion_sites_qt, cells)

def regulate(promotors, diffusion_sites_qt, cells):
    maternal_injection(promotors, diffusion_sites_qt, cells)
    growth()

def maternal_injection(promotors, diffusion_sites_qt, cells):
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
    mother_tf_label = promotors[first_gene_idx][tf_label_idx]
    # Get minimum amount of regulatory tf to regulate regulated product
    mother_tf_injection = float(promotors[first_gene_idx][min_value_idx])

    # Create first cell
    first_cell = Cell()

    # Distributes injection among diffusion sites
    first_cell.transcription_factors[mother_tf_label] = \
        [mother_tf_injection / diffusion_sites_qt] * diffusion_sites_qt
    
    # .....
    self.express_promoters(first_cell)
    
    # Append first cell
    cells.append(first_cell)

    # Develop a module based on it
    first_cell.developed_module = self.place_head(first_cell)

class Cell:
    """Goal:
        Class to model a cell.
    -----------------------------------------------------
    Input:
        self: object"""

    def __init__(self) -> None:
        self.developed_module = None
        self.transcription_factors = {}


def develop(querying_seed: int) -> BodyV2:

    # Initialize
    rng = random.Random(querying_seed) # Get random number generator
    part_count = 0 # Number of body parts

    # Call functions
    develop_body()


def gene_parser(promotors, genotype, promoter_threshold, types_nucleotypes, regulatory_transcription_factor_idx,
                regulatory_min_idx, regulatory_max_idx, transcription_factor_idx, transcription_factor_amount_idx,
                diffusion_site_idx, structural_trs, regulatory_tfs, diffusion_sites_qt):
    """Goal:
        Create genes from the genotype.
    -----------------------------------------------------------------------------------------------
    Input:
        ...
    -----------------------------------------------------------------------------------------------
    Output:
        ..."""
    
    
    # Initialize nucleotide index
    nucleotide_idx = 0

    # Repeat as long as index is smaller than gene length
    while nucleotide_idx < len(genotype):
        # If the associated value is smaller than the promoter threshold
        if genotype[nucleotide_idx] < promoter_threshold:
            # If there are nucleotypes enough to compose a gene
            if (len(genotype) - 1 - nucleotide_idx) >= types_nucleotypes:
                # Get regulatory transcription factor(s)
                regulatory_transcription_factor = genotype[nucleotide_idx + regulatory_transcription_factor_idx + 1]  # Gene product
                regulatory_min = genotype[nucleotide_idx + regulatory_min_idx + 1]
                regulatory_max = genotype[nucleotide_idx + regulatory_max_idx + 1]
                # Get transcription factor, -amount and diffusion site
                transcription_factor = genotype[nucleotide_idx + transcription_factor_idx + 1]
                transcription_factor_amount = genotype[nucleotide_idx + transcription_factor_amount_idx + 1]
                diffusion_site = genotype[nucleotide_idx + diffusion_site_idx + 1]
                
                # Converts tfs values into labels
                range_size = 1 / (structural_trs + regulatory_tfs)
                limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                for idx in range(0, len(limits) - 1): # Why up to minus 1?????
                    # Set label for regulatory transcription factor
                    if (regulatory_transcription_factor >= limits[idx]) and (regulatory_transcription_factor < limits[idx + 1]):
                        regulatory_transcription_factor_label = 'TF' + str(idx + 1)
                    elif regulatory_transcription_factor >= limits[idx + 1]: # ??????
                        regulatory_transcription_factor_label = 'TF' + str(len(limits))
                    # Set label for transcription factor
                    if (transcription_factor >= limits[idx]) and (transcription_factor < limits[idx + 1]):
                        transcription_factor_label = 'TF' + str(idx + 1)
                    elif transcription_factor >= limits[idx + 1]: # ???????
                        transcription_factor_label = 'TF' + str(len(limits))
        
                # Converts diffusion sites values into labels
                range_size = 1 / diffusion_sites_qt
                limits = [round(limit / 100, 2) for limit in range(0, 1 * 100, int(range_size * 100))]
                for idx in range(0, len(limits) - 1): # Why up to minus 1?????
                    if limits[idx+1] > diffusion_site >= limits[idx]:
                        diffusion_site_label = idx
                    elif diffusion_site >= limits[idx + 1]: # ???????
                        diffusion_site_label = len(limits) - 1
                
                # Create gene
                gene = [regulatory_transcription_factor_label, regulatory_min, regulatory_max,
                            transcription_factor_label, transcription_factor_amount, diffusion_site_label]

                # Append gene to promoters
                promotors.append(gene)

                # Increase nucleotide index
                nucleotide_idx += types_nucleotypes
        
        # Increase nucleotide index
        nucleotide_idx += 1
    
    # Convert to numpy
    promotors = np.array(promotors)

    return promotors
    