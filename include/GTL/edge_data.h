#ifndef GTL_EDGE_DATA_H
#define GTL_EDGE_DATA_H

#include <GTL/GTL.h>
#include <GTL/node.h>
#include <GTL/edge.h>
#include <GTL/graph.h>

#include <list>
#include <map>

__GTL_BEGIN_NAMESPACE

/**
 * @internal
 */
class GTL_EXTERN edge_data
{
public:
    int id;				// internal numbering
	nodes_t nodes[2]; 		// nodes[0] = sources,
    					// nodes[1] = targets
	std::list<edges_t::iterator> adj_pos[2];// positions in the adjacency lists
					// of sources and targets
	edges_t::iterator pos;		// position in the list of all edges
    bool hidden;
    graph* owner;
};

__GTL_END_NAMESPACE

#endif // GTL_EDGE_DATA_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
