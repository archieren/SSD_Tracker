#ifndef GTL_NODE_DATA_H
#define GTL_NODE_DATA_H

#include <GTL/GTL.h>
#include <GTL/node.h>
#include <GTL/edge.h>

#include <list>
#include <map>

//node_data.h - Internal header: DO NO USE IT DIRECTLY !!!
__GTL_BEGIN_NAMESPACE

class graph;

/**
 * @internal
 */
class GTL_EXTERN node_data
{
public:
    int id;			// internal numbering
    graph *owner;		// graph containing this node
    nodes_t::iterator pos;	// position in the list of all nodes
	edges_t edges[2];	// edges incident to this node
				// edges[0] = in_edges, edges[1] = out_edges
    bool hidden;
};

__GTL_END_NAMESPACE

#endif // GTL_NODE_DATA_H

