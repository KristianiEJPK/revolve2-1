import sqlalchemy.orm as orm

class BodyMappingSeedOrmV2(orm.MappedAsDataclass, kw_only=True):
    """SQLAlchemy model for a CPPNWIN body genotype. What the function
        basically does is setting the mapping seed as a column in the
        genotype table."""
    mapping_seed: orm.Mapped[int] = orm.mapped_column(nullable=False)
    
