#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Docstrings are good! */
static char module_docstring[] = "Demo for astropy.";

static char convolve1d_boundary_wrap_docstring[] =
                "Convolve a little.";

/* Declare the C functions here. */
static PyObject *convolution_convolve1d_boundary_wrap(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"convolve1d_boundary_wrap", convolution_convolve1d_boundary_wrap, METH_VARARGS, convolve1d_boundary_wrap_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */
PyMODINIT_FUNC init_convolution(void)
{
    /* Initialize the module with a docstring. */
    PyObject *m = Py_InitModule3("_convolution", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load all of the `numpy` functionality. */
    import_array();
}

/* Do the heavy lifting here */
static PyObject *convolution_convolve1d_boundary_wrap(PyObject *self, PyObject *args)
{
    PyObject *f_obj, *g_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &f_obj, &g_obj))
        return NULL;

    /* Interpret the input objects as `numpy` arrays. */
    PyObject *f_array = PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *g_array = PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (f_array == NULL || g_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(f_array);
        Py_XDECREF(g_array);
        return NULL;
    }

    /* Check that the input arrays are 1D. */
    int f_dim = (int)PyArray_NDIM(f_array);
    int g_dim = (int)PyArray_NDIM(g_array);
    if (f_dim != 1 || g_dim != 1) {
        PyErr_SetString(PyExc_TypeError, "The input arrays must be 1D.");
        Py_DECREF(f_array);
        Py_DECREF(g_array);
        return NULL;
    }

    /* Get the lengths of the outputs. */
    int nx = (int)PyArray_DIM(f_array, 0);
    int nkx = (int)PyArray_DIM(g_array, 0);

    if (nkx % 2 != 1) {
        PyErr_SetString(PyExc_TypeError, "Convolution kernel must have odd dimensions.");
        Py_DECREF(f_array);
        Py_DECREF(g_array);
        return NULL;
    }

    /* Build the output array. */
    npy_intp dims[1];
    dims[0] = nx;
    PyObject *out_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (out_array == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array.");
        Py_DECREF(f_array);
        Py_DECREF(g_array);
        Py_XDECREF(out_array);
        return NULL;
    }
    double *out = (double*)PyArray_DATA(out_array);

    /* Allocate memory for the `fixed` array. */
    double *fixed = (double*)malloc(nx * sizeof(double));

    /* Get pointers to the data as C-types. */
    double *f = (double*)PyArray_DATA(f_array);
    double *g = (double*)PyArray_DATA(g_array);

    /* Define some variables. */
    int i, ii, iii, iimin, iimax;
    double top, bot, val, ker;
    int wkx = nkx / 2;

    /*
        Need a first pass to replace NaN values with value convolved from
        neighboring values.
    */
    for (i = 0; i < nx; i++) {
        if (npy_isnan(f[i]) != 0) {
            top = 0.0;
            bot = 0.0;
            iimin = i - wkx;
            iimax = i + wkx + 1;
            for (ii = iimin; ii < iimax; ii++) {
                if (ii < 0) iii = (nx + ii) % nx;
                else iii = ii % nx;
                val = f[iii];
                if (npy_isnan(val) == 0) {
                    ker = g[wkx + ii - i];
                    top += val * ker;
                    bot += ker;
                }
            }

            if (bot != 0.0) {
                fixed[i] = top / bot;
            } else {
                fixed[i] = f[i];
            }
        } else {
            fixed[i] = f[i];
        }
    }

    /* Now run the proper convolution. */
    for (i = 0; i < nx; i++) {
        if (npy_isnan(fixed[i]) == 0) {
            top = 0.0;
            bot = 0.0;
            iimin = i - wkx;
            iimax = i + wkx + 1;
            for (ii = iimin; ii < iimax; ii++) {
                if (ii < 0) iii = (nx + ii) % nx;
                else iii = ii % nx;
                val = fixed[iii];
                if (npy_isnan(val) == 0) {
                    ker = g[wkx + ii - i];
                    top += val * ker;
                    bot += ker;
                }
            }

            if (bot != 0.0) {
                out[i] = top / bot;
            } else {
                out[i] = fixed[i];
            }
        } else {
            out[i] = fixed[i];
        }
    }

    /* Clean up. */
    free(fixed);
    Py_DECREF(f_array);
    Py_DECREF(g_array);

    return out_array;
}
