cmake_minimum_required(VERSION 3.5)

# ----------------------------------------------------------------------------
# Set Project Name
# ----------------------------------------------------------------------------
#set (PROJECT_NAME  SSD)
project(SSD)

# ----------------------------------------------------------------------------
# Find Libraries
# ----------------------------------------------------------------------------
# OpenCV >= 3.4
FIND_PACKAGE(OpenCV REQUIRED)
find_package(TensorFlow REQUIRED)
# ----------------------------------------------------------------------------
# Add Libraries and peoject files
# ----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
include_directories(${TensorFlow_INCLUDE_DIRS})

INCLUDE(CheckIncludeFileCXX)

INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/include
        )

# ----------------------------------------------------------------------------
# Add Definition and Options
# ----------------------------------------------------------------------------
option(USE_OPENCV "Build with OpenCV support" ON)

# ----------------------------------------------------------------------------
# SELECT C++ standard
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)

# ----------------------------------------------------------------------------
# Link source and header files
# ----------------------------------------------------------------------------
set(graph_source
        src/graph/tokenise.cpp
        src/graph/mytree.cpp
        src/graph/mygraph.cpp
        src/graph/mwbmatching.cpp
        src/graph/mincut.cpp
        src/graph/gport.cpp
        src/graph/fheap.c
        )

set(graph_header
        include/graph/tokenise.h
        include/graph/mytree.h
        include/graph/mygraph.h
        include/graph/mwbmatching.h
        include/graph/mincut.h
        include/graph/gport.h
        include/graph/gdefs.h
        include/graph/fheap.h
        )

set(gtl_source
        src/GTL/bellman_ford.cpp
        src/GTL/bfs.cpp
        src/GTL/biconnectivity.cpp
        src/GTL/bid_dijkstra.cpp
        src/GTL/components.cpp
        src/GTL/debug.cpp
        src/GTL/dfs.cpp
        src/GTL/dijkstra.cpp
        src/GTL/edge.cpp
        src/GTL/embedding.cpp
        src/GTL/fm_partition.cpp
        src/GTL/gml_parser.cpp
        src/GTL/gml_scanner.cpp
        src/GTL/graph.cpp
        src/GTL/maxflow_ff.cpp
        src/GTL/maxflow_pp.cpp
        src/GTL/maxflow_sap.cpp
        src/GTL/min_tree.cpp
        src/GTL/node.cpp
        src/GTL/planarity.cpp
        src/GTL/pq_node.cpp
        src/GTL/pq_tree.cpp
        src/GTL/ratio_cut_partition.cpp
        src/GTL/st_number.cpp
        src/GTL/topsort.cpp
        )

set(gtl_header
        include/GTL/algorithm.h
        include/GTL/bellman_ford.h
        include/GTL/bfs.h
        include/GTL/biconnectivity.h
        include/GTL/bid_dijkstra.h
        include/GTL/bin_heap.h
        include/GTL/components.h
        include/GTL/debug.h
        include/GTL/dfs.h
        include/GTL/dijkstra.h
        include/GTL/edge.h
        include/GTL/edge_data.h
        include/GTL/edge_map.h
        include/GTL/embedding.h
        include/GTL/fm_partition.h
        include/GTL/gml_parser.h
        include/GTL/gml_scanner.h
        include/GTL/graph.h
        include/GTL/maxflow_ff.h
        include/GTL/maxflow_pp.h
        include/GTL/maxflow_sap.h
        include/GTL/min_tree.h
        include/GTL/ne_map.h
        include/GTL/node.h
        include/GTL/node_data.h
        include/GTL/node_map.h
        include/GTL/planarity.h
        include/GTL/pq_node.h
        include/GTL/pq_tree.h
        include/GTL/ratio_cut_partition.h
        include/GTL/st_number.h
        include/GTL/symlist.h
        include/GTL/topsort.h
        include/GTL/version.h
        include/GTL/GTL.h
        )

set(tracker_headers
        include/tracker/Ctracker.h
        include/tracker/track.h
        include/tracker/LocalTracker.h
        include/tracker/Kalman.h
        include/HungarianAlg/HungarianAlg.h
        )
set(tracker_source
        src/tracker/Ctracker.cpp
        src/tracker/track.cpp
        src/tracker/LocalTracker.cpp
        src/tracker/Kalman.cpp
        src/HungarianAlg/HungarianAlg.cpp        
        )

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/ssd_detect.cpp
        include/ssd_detect.h
        include/defines.h
        include/nms.h
        ${tracker_source} ${tracker_headers} ${graph_source} ${graph_header} ${gtl_source} ${gtl_header}
)

SOURCE_GROUP("Source Files" FILES ${tracker_source})
SOURCE_GROUP("Header Files" FILES ${tracker_headers})

SOURCE_GROUP("graph" FILES ${graph_source} ${graph_header})
SOURCE_GROUP("GTL" FILES ${gtl_source} ${gtl_header})

# ----------------------------------------------------------------------------
# Link libraries to project
# ----------------------------------------------------------------------------
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} ${TensorFlow_LIBS})
