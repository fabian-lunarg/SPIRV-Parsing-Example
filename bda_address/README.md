# Buffer Device Address

If you have a shader such as

```glsl
layout(buffer_reference, scalar) buffer Indices {
    uint i[];
};

struct GeometryNode {
	uint64_t address;
};
layout(binding = 4, set = 1) buffer GeometryNodes {
    GeometryNode nodes[];
} geometryNodes;

void SomeLogic(int index) {
	GeometryNode geometryNode = geometryNodes.nodes[index];

	Indices indices = Indices(geometryNode.address);

    indices.i[0] = 0;
}
```

This pass will help detect when where the `address` value came from.

For example at `indices.i[0] = 0;` there is a `OpLoad` where dereference the pointer at `address`.

From there we work back how we got it and detect that the "address" was from the buffer in `binding 4, set 1`