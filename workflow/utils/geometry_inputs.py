"""Keep visual answer candidates out of reconstruction without hiding them from VQA."""
from dataclasses import replace

def select_geometry_images(images, indices=None):
    if indices is None:return images
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or i >= len(images) for i in indices):
        raise ValueError('Invalid geometry image indices')
    return [images[i] for i in indices]

def geometry_view(images, spec, indices=None):
    selected=select_geometry_images(images, indices)
    if indices is None:return selected,spec
    if spec.reference_view-1 not in indices:raise ValueError('Reference view is an answer candidate, not a reconstruction observation')
    return selected,replace(spec,reference_view=indices.index(spec.reference_view-1)+1)
