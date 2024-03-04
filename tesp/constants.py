"""Constants
---------.
"""

from typing import ClassVar

from wagl.constants import ArdProducts


class ProductPackage:
    """Helper class for selecting which ard products to package."""

    _default_excludes: ClassVar[set[str]] = {
        ArdProducts.LAMBERTIAN.value.lower(),
        ArdProducts.SBT.value.lower(),
    }

    _all_products: ClassVar[set[str]] = {e.value.lower() for e in ArdProducts}

    @classmethod
    def validate_products(cls, product_list) -> bool:
        return set(product_list).issubset(cls._all_products)

    @classmethod
    def all(cls) -> set[str]:
        return cls._all_products

    @classmethod
    def default(cls) -> set[str]:
        return cls._all_products - cls._default_excludes
