#ifndef GTL_GTL_H
#define GTL_GTL_H

#include <GTL/version.h>

//--------------------------------------------------------------------------
//   Generic iteration over container elements
//--------------------------------------------------------------------------
//
// elem: loop variable
// cont: container to iterate over
// iter_t: iterator type
// iter: prefix for begin() and end()
//
// contains a hack for Microsoft Visual C++ 5.0, because code like
//
//   for(int i=0; i<10; ++i) { ... do something ... }
//   for(int i=0; i<10; ++i) { ... do something again ... }
//
// is illegal with Microsoft Extensions enabled, but without Microsoft
// Extensions, the Microsoft STL does not work :-(.
// So we code the line number (__LINE__) into our loop variables.

#define GTL_CONCAT(x, y) x##y
#define GTL_FORALL_VAR(y) GTL_CONCAT(GTL_FORALL_VAR, y)

#define GTL_FORALL(elem, cont, iter_t, iter)			\
if ((cont).iter##begin() != (cont).iter##end())			\
    (elem) = *((cont).iter##begin());				\
for (iter_t GTL_FORALL_VAR(__LINE__) = (cont).iter##begin();    \
    GTL_FORALL_VAR(__LINE__) != (cont).iter##end();             \
    (elem) = (++GTL_FORALL_VAR(__LINE__)) ==                    \
        (cont).iter##end() ? (elem) : *GTL_FORALL_VAR(__LINE__))

#define __GTL_USE_NAMESPACES
#define GTL_EXTERN


#ifdef __GTL_USE_NAMESPACES
#  define __GTL_BEGIN_NAMESPACE namespace GTL {
#  define __GTL_END_NAMESPACE }
#else
#  define __GTL_BEGIN_NAMESPACE
#  define __GTL_END_NAMESPACE
#endif

#ifdef __GTL_USE_NAMESPACES
namespace GTL {}
using namespace GTL;
#endif // __GTL_USE_NAMESPACES

#endif // GTL_GTL_H

//--------------------------------------------------------------------------
//   end of file
//--------------------------------------------------------------------------
