# Standard library imports 
import os.path, glob

# Major library imports  
import pylab as pylab
import numpy as np

# Enthought imports 
from enthought.traits.api \
    import HasTraits, Range, Float, Enum, Instance, ListFloat
from enthought.traits.trait_numeric \
    import Array
from enthought.traits.ui.api \
    import View, Group, VGroup, HGroup, Item, \
    ScrubberEditor, spring
from enthought.traits.ui.ui_traits \
    import ATheme

from enthought.enable.api \
    import Component, ComponentEditor

from enthought.chaco.api \
    import Plot, OverlayPlotContainer, GridPlotContainer, \
    ArrayPlotData, GridContainer, ImageData, ColorMapper, \
    gmt_drywet
from enthought.chaco.array_data_source import ArrayDataSource
from enthought.chaco.shell \
    import imshow, title, show
from enthought.chaco.tools.api \
    import PanTool, ZoomTool
import enthought.chaco.default_colormaps as chaco_colormaps
from enthought.traits.ui.menu import Action, CloseAction, Menu, \
                                     MenuBar, NoButtons, Separator

# Nipy imports
from nipy.io.api import load_image
from nipy.algorithms.resample import resample
from nipy.core.api import Image
import nipy.algorithms.registration as R

# Scipy imports
from scipy.io import loadmat

# Local library imports
from brain.paths import mask_full, anat, anat_hires
from brain.cv import Kfold_available, CV
from brain.data import unmask, MASK_FULL
from brain.get_results import test_data, _get_results # _get_median_rates
#from gui.rate_explorer import PlotUI, Model, Controller, ModelView

pjoin = os.path.join

# to be moved into input class
results_base = '/media/500A/Results/GraphNet/resubmission/CV5/'
filename = 'NaiveGraphNet5.mat'
model_type = 'NaiveGraphNet'
#cv_type = 'CV5'
#K = 5

cdict_default = {'red':   ((0,    0,   0),
                           (0,    0,   0),
                           (0.5,  0.1, 0.1),
                           (0.85, 1.0, 1.0),
                           (1.0,  1.0, 1.0)),
                 'green': ((0,    1,    1),
                           (0.15, 0.5,  0.5),
                           (0.5,  0.,   0.),
                           (0.7,  0.,   0.),
                           (0.85, 0.85, 0.85),
                           (1.0,  1.0,  1.0)),
                 'blue':  ((0,    1,    1),
                           (0.40, 0.5,  0.5), 
                           (0.5,  0,    0),
                           (0.85, 0.,   0.),
                           (1.0,  1.,  1.)),
                 'alpha': ((0.,   1.,   1.), 
                           (0.4,  0.,   1.),
                           (0.496,0.,   0.),
                           (0.5,  0.,   0.),
                           (0.504,1.,   0.),
                           (0.6,  1.,  1.),
                           (1.0,  1.,  1.)) }

my_cmap = pylab.matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict_default,256)

#rate_list, rate_array, rate_lut =_get_median_rates( pjoin( results_base, model_type + cv_type ) )

rate_list, rate_array, rate_lut, param_arr, coef_sparr, best_fit, best_folds =_get_results( pjoin( results_base, filename ), median = True, mat = True, legacy = False )

#1/0

if model_type is 'GraphSVM':
    coef_sparr = coef_sparr[:, 1::]

#rate_list, rate_array, rate_lut, param_arr, coef_sparr, best_fit, best_folds =_get_results( pjoin( results_base, filename ), median = False, mat = True, legacy = True )
#1/0

class ThemedItem ( Item ):
    editor     = ScrubberEditor( hover_color  = 0xFFFFFF )
    item_theme = ATheme( '@std:LG' )

class Models( HasTraits ):

    x_coord        = Range( -30, 30 ,
                             desc = "The X-slice to be displayed.",
                             label = "X coordinate" )
    
    y_coord        = Range( -30, 30 ,
                             desc = "The Y-slice to be displayed.",
                             label = "Y coordinate" )
    
    z_coord        = Range( -20, 20 ,
                             desc = "The Z-slice to be displayed.",
                             label = "Z coordinate" )
    
    t_coord        = Range( 0, 6 ,
                            desc = "The time point to be displayed.",
                            label = "Time point" )
    
    lambda_1       = Range( float( min(rate_lut['l1']) ), float( max(rate_lut['l1']) ) ,
                            desc = "The l1-penalty ('LASSO') parameter.",
                            label = "lambda 1" )
    
    lambda_2       = Range( float( min(rate_lut['l2']) ), float( max(rate_lut['l2']) ) ,
                            desc = "The l2-penalty ('Ridge') parameter.",
                            label = "lambda 2" )
    
    lambda_G       = Range( float( min(rate_lut['lG']) ), float( max(rate_lut['lG']) ) ,
                            desc = "The graph-penalty parameter.",
                            label = "lambda G" )
    
    delta          = Range( float( min( rate_lut['delta'] ) ), float( max( rate_lut['delta'] ) ),
                            desc = "The Huber loss parameter.",
                            label = "Huber delta" )
    
    lambda_1_star  = Range( float( min(rate_lut['l1_star']) ), float( max(rate_lut['l1_star']) ),
                            desc = "The Adaptive l1-penalty parameter.",
                            label = "Adaptive lambda 1" )
    
    penalty_type   =  Enum( "LASSO","ENet","Spatial Graph Laplacian","Spatialtemporal Graph Laplacian","Adaptive GraphNet",
                            desc = "The type of coefficient penalization used in fitting the model.",
                            label = "Model penalty type" )
    
    loss_type      =  Enum( "Least-Squares","Robust (Huber)","Maximum Margin (SVM classification)",
                            desc = "The type of loss function used in fitting the model",
                            label = "Model Loss Function type." )
    
    plot_data      = ArrayPlotData()

    rate_data      = ArrayPlotData()

    rate           = Float( 0.,
                            desc = "Test error from cross-validation.",
                            label = "Cross-validated classification accuracy" )

    def __init__( self ):

        self.dim     = len(rate_array.shape)
        self.params  = {'l1':      self.lambda_1, 
                        'l2':      self.lambda_2, 
                        'lG':      self.lambda_G,
                        'delta':   self.delta,
                        'l1_star': self.lambda_1_star
                        }

        self.coefs      = coef_sparr
        self.rate_array = rate_array
        self.rate_lut   = rate_lut 
        self.param_arr  = param_arr

        self.anatim = load_image( anat )
        self.anathi = load_image( anat_hires )

        self.affine_resample = None

        self._update_params()

        self.rate_data.set_data( 'rates', self.rates.astype(int) )        

        
    def fig_slice( self, 
                   axis     = 0, 
                   coords   = None,
                   outfile  = None,
                   plotit   = False,
                   upsample = True ):
        
        """                                                                                                   
        Plots or returns a brain slice and coefficients for the given axis and coordinate vector.                                                  
        """
        if coords is None:
            coords = [self.x_coord, self.y_coord, self.z_coord, self.t_coord]

        if upsample is True:
            anat = self.anathi
        else:
            anat = self.anatim

        brainslice = ( int( anat.coordmap.inverse()( coords[0:3] )[0] ),
                       int( anat.coordmap.inverse()( coords[0:3] )[1] ),
                       int( anat.coordmap.inverse()( coords[0:3] )[2] ) )

        im          = self.im
        self.im_max = np.max( im )

        if axis is 0:
            a = np.asarray( anat )[ brainslice[0], :, : ].T
            i = im[ brainslice[0], :, : ].T
        elif axis is 1:
            a = np.asarray( anat )[ :, brainslice[1], : ].T
            i = im[ :, brainslice[1], : ].T
        else:
            a = np.asarray( anat )[ :, :, brainslice[2] ].T
            i = im[ :, :, brainslice[2] ].T

        if plotit is True:
            pylab.clf()

            pylab.imshow( a, 
                          cmap          = pylab.cm.gray, 
                          interpolation = 'nearest', 
                          origin        = 'lower' )

            pylab.imshow( self._nan_thresh( i ),
                          interpolation = 'nearest', 
                          origin        = 'lower',
                          vmin          = -abs( im[brainslice].T ).max(), 
                          vmax          =  abs( im[brainslice].T ).max(), 
                          cmap          = my_cmap )

            pylab.colorbar()

            pylab.gca().set_xticks([])
            pylab.gca().set_yticks([])
            
            if not outfile:
                pylab.show()
            else:
                pylab.savefig(outfile)
        else:
            return a, 255 * self._scale_coefs( i ) 

    def _set_coef_block( self,
                         coords   = None,
                         mask     = MASK_FULL, 
                         upsample = True ):

        """
        Gets a coefficient block corresponding to a set of model parameters.
        """

        if coords is None:
            coords = [self.x_coord, self.y_coord, self.z_coord, self.t_coord]

        coefs = self._get_coefs( self.params ) 

        ims = unmask( coefs, mask )
        im  = ims[ coords[3] ] / np.fabs( ims[ coords[3] ] ).max()
        im  = Image( im, self.anatim.coordmap )

        if upsample == True:
            print 'resampling'

            if self.affine_resample is None:

                h      = R.HistogramRegistration( im, self.anathi )
                self.affine_resample = h.optimize( 'affine' ).as_affine()

            resampled_im = resample( im,
                                     self.anathi.coordmap,
                                     self.affine_resample,
                                     self.anathi.shape,
                                     order = 1 )

#            reim  = resample( im, self.anathi.coordmap, np.identity(4), self.anathi.shape, order=1 )
            print 'done'
#            im = np.ma.masked_array( np.asarray(reim), np.less( np.fabs(reim), 0.05 ) )
            im = np.asarray( resampled_im )
            self.im = im


    def rate_slice( self, slice_select = None ):
                    
        idx = [ self._get_index_lut( self.rate_lut['l1'],      self.lambda_1 ),
                self._get_index_lut( self.rate_lut['l2'],      self.lambda_2 ),
                self._get_index_lut( self.rate_lut['lG'],      self.lambda_G ),
                self._get_index_lut( self.rate_lut['delta'],   self.delta    ),
                self._get_index_lut( self.rate_lut['l1_star'], self.lambda_1_star ) ]
        
        self.rate = self.rate_array[ idx[0:self.dim] ][0]

        if slice_select is None:
            slice_select = [ True, True, False, False, False ]

        for i in range( len(slice_select) ):
            if slice_select[i] is True:
                idx[i] = Ellipsis

        self.idx  = idx

        self.rates = 256 * np.squeeze( self.rate_array[ idx[0:self.dim] ] )


    def _get_coefs( self, params ):
#        print params
#        if   len( [params['delta']] ) is 1:
#            return [ v for k,v in self.coef_dict.iteritems() if k  == ( params['l1'],
#                                                                        params['l2'],
#                                                                        params['lG'], 0, 0 ) ][0]
#        elif len( [params['delta']] ) is not 1 and len( [params['l1_star']] ) is 1:
#            return [ v for k,v in self.coef_dict.iteritems() if k  == ( params['l1'],
#                                                                        params['l2'],
#                                                                        params['lG'],
#                                                                        params['delta'], 0 ) ][0]
#        elif len( params[['l1_star']] ) is not 1:

        if self.dim == 1:
            current_params = ( params['l1'] )
        elif self.dim == 3:
            current_params = ( params['l1'],
                               params['l2'],
                               params['lG'] )
        elif self.dim == 4:
            current_params = ( params['l1'],
                               params['l2'],
                               params['lG'],
                               params['delta'])
        else:
            current_params = ( params['l1'],
                               params['l2'],
                               params['lG'],
                               params['delta'],
                               params['l1_star'] ) 

        for i in range( self.param_arr.shape[0] ):

            if self.dim == 1:
                candidate_params = ( self.param_arr[i,0] )
            elif self.dim == 3:
                candidate_params = ( self.param_arr[i,0],
                                     self.param_arr[i,1],
                                     self.param_arr[i,2] )
            elif self.dim == 4:
                candidate_params = ( self.param_arr[i,0],
                                     self.param_arr[i,1],
                                     self.param_arr[i,2],
                                     self.param_arr[i,3] )
            else:
                candidate_params = ( self.param_arr[i,0],
                                     self.param_arr[i,1],
                                     self.param_arr[i,2],
                                     self.param_arr[i,3],
                                     self.param_arr[i,4] )

            if candidate_params == current_params:
                idx = i

        return self.coefs[idx,:].todense()

    def _scale_coefs( self, coefs ):
        if np.sum( np.fabs( coefs ) ) > 0:
            return coefs / np.max( np.fabs( coefs ) ) 
        else:
            return coefs

    def _get_index_lut( self, lut, val ):
        return np.where( np.asarray( lut )  == np.array( val ) )[0]

    def _nan_thresh( self, x, thresh = 0.05 ):
        s = x.shape
        out = x.copy()
        for i in range( s[0] ):
            for j in range( s[1] ):
                if np.abs(out[i,j]) <= thresh: 
                    out[i,j] = float('nan')
        return out

    def _zero_thresh( self, x, thresh = 0.00005 ):
        s = x.shape
        out = x.copy()
        for i in range( s[0] ):
            for j in range( s[1] ):
                if np.abs(out[i,j]) <= thresh: 
                    out[i,j] = 0.
        return out

    def _update_params( self ):
        self.params  = {'l1':      self.lambda_1,
                        'l2':      self.lambda_2,
                        'lG':      self.lambda_G,
                        'delta':   self.delta,
                        'l1_star': self.lambda_1_star
                        }
        self._get_set_nearest_params()
        self._set_coef_block()
        self.rate_slice()
        self.sagittal()
        self.coronal()
        self.axial()        

    def _get_set_nearest_params( self ):
        """ Get closest actual parameter values to slider values and set values to these."""
        for k in self.params.keys():
            idx = np.argmin( np.fabs( np.asarray(self.rate_lut[k]) - self.params[k] ) )
            print idx
            if len( self.rate_lut[k] ) == 1:
                self.params[k] = self.rate_lut[k]
            else:
                self.params[k] = self.rate_lut[k][ idx ]

    def axial( self ):
        self.axial_anat, self.axial_coefs = self.fig_slice(axis   = 2, 
                                    coords = [self.x_coord, self.y_coord, self.z_coord, self.t_coord] )
        self.plot_data.set_data( "axial_anat", self.axial_anat.astype(int) ) 
        self.plot_data.set_data( "axial_coefs", self.axial_coefs.astype(int) ) 


    def coronal( self ):
        self.coronal_anat, self.coronal_coefs = self.fig_slice(axis   = 1,
                                    coords = [self.x_coord, self.y_coord, self.z_coord, self.t_coord] )
        self.plot_data.set_data( "coronal_anat", self.coronal_anat.astype(int) ) 
        self.plot_data.set_data( "coronal_coefs", self.coronal_coefs.astype(int) ) 

    def sagittal( self ):
        self.sagittal_anat, self.sagittal_coefs = self.fig_slice(axis   = 0,
                                    coords = [self.x_coord, self.y_coord, self.z_coord, self.t_coord] )
        self.plot_data.set_data( "sagittal_anat", self.sagittal_anat.astype(int) ) 
        self.plot_data.set_data( "sagittal_coefs", self.sagittal_coefs.astype(int) ) 

    def _x_coord_changed( self ):
        self.sagittal()

    def _y_coord_changed( self ):
        self.coronal()

    def _z_coord_changed( self ):
        self.axial()

    def _t_coord_changed( self ):
        self._set_coef_block()
        self.sagittal()
        self.coronal()
        self.axial()
        # add threading?

    def _lambda_1_changed( self ):
        print "woot woot"
        self._update_params()

    def _lambda_2_changed( self ):
        self._update_params()

    def _lambda_G_changed( self ):
        self._update_params()

    def _delta_changed( self ):
        self._update_params()

    def _lambda_1_star_changed( self ):
        self._update_params()

    def _penalty_type_changed( self ):

        if self.penalty_type is "Lasso":
            self.lambda_2 = 0.
            self.lambda_G = 0.
        elif self.penalty_type is "ENet":
            self.lambda_G = 0.
        else:
            self._loss_type_changed()

    def _loss_type_changed( self ):

        if self.loss_type is "Least-Squares":
            filename = 'NaiveGraphNet.mat'
        elif self.loss_type is "Robust (Huber)" and self.penalty_type is "Spatial GraphNet":
            filename = 'RobustGraphNet.mat'
        elif self.loss_type is "Robust (Huber)" and self.penalty_type is "Spatiotemporal GraphNet":
            filename = 'RobustGraphNet_time.mat'
        elif self.loss_type is "Robust (Huber)" and self.penalty_type is "Adaptive GraphNet":
            filename = 'RobustGraphNetReweight.mat'
        elif self.loss_type is "Maximum Margin (SVM classification)":
            filename = 'GraphSVM.mat'
        else:
            print self.loss_type
            filename = 'RobustGraphNet.mat'
#            raise NotImplementedError()


#        rate_list, rate_array, rate_lut, param_arr, coef_sparr =_get_median_results( pjoin( results_base, filename ), mat = True )

        self.dim     = len(rate_array.shape)
        self.params  = {'l1':      self.lambda_1, 
                        'l2':      self.lambda_2, 
                        'lG':      self.lambda_G,
                        'delta':   self.delta,
                        'l1_star': self.lambda_1_star
                        }

        self.coefs      = coef_sparr
        self.rate_array = rate_array
        self.rate_lut   = rate_lut 
        self.param_arr  = param_arr

        self._update_params()

        self.rate_data.set_data( 'rates', self.rates.astype(int) )        


class MRIBrainContainer( Models ):

    traits_view = View( Item( 'plot', 
                              editor     = ComponentEditor(), 
                              show_label = False,
                              width      = 1200,
                              height     = 1200,
                              resizable  = True,
                              ),
                        kind = 'live',
                        title = 'Neuroparser'
                        )

    def __init__( self, **traits ):
        super( MRIBrainContainer, self ).__init__( **traits )
        
        self.cmap =  chaco_colormaps.center( self.default_cmap )

        sp11 = self.slice_plot( "coronal_anat",  "coronal_coefs"  )
        sp12 = self.slice_plot( "sagittal_anat", "sagittal_coefs" )
        sp21 = self.slice_plot( "axial_anat",    "axial_coefs"    )
        sp22 = self.rate_plot ( "rates" ) 

        container = GridContainer( shape = (2,2),
                                   padding = 20,
                                   fill_padding = True,
                                   bgcolor = "lightgray",
                                   use_backbuffer = True )

        container.add( sp11 ) 
        container.add( sp12 ) 
        container.add( sp21 ) 
        container.add( sp22 ) 

        self.plot = container

    def slice_plot( self, anat, coefs, **traits ):

        p  = Plot( self.plot_data, default_origin = 'bottom left' )
        p2 = Plot( self.plot_data, default_origin = 'bottom left' )

        p.x_axis.visible = False; p2.x_axis.visible = False
        p.y_axis.visible = False; p2.y_axis.visible = False

        bounds = self.plot_data.get_data(anat).shape
        asp    = float( bounds[1] ) / float( bounds[0] )

        p.img_plot( anat,
#                    xbounds  = np.linspace( 0, 1, bounds[1] + 1 ),
#                    ybounds  = np.linspace( 0, 1, bounds[0] + 1 ),
                    colormap = chaco_colormaps.gray )
        
        p2.img_plot( coefs,
#                     xbounds  = np.linspace( 0, 1, bounds[1] + 1 ),
#                     ybounds  = np.linspace( 0, 1, bounds[0] + 1 ),
#                     bgcolor = 'transparent',
                     colormap = self.cmap,
                     interpolation = 'nearest')

#        p.aspect_ratio = asp; p2.aspect_ratio = asp
        p.aspect_ratio = asp; p2.aspect_ratio = asp

        subplot = OverlayPlotContainer( )

        subplot.add( p )
        subplot.add( p2 )

        return subplot 

    def rate_plot( self, rates, **traits ):

        rp = Plot( self.rate_data, default_origin = 'bottom left' )

        rp.x_axis.visible = False 
        rp.y_axis.visible = False 

#        bounds = self.rate_data.get_data( rates ).shape

        rp.img_plot( rates,
#                     xbounds  = np.linspace( 0, 1, bounds[1] + 1 ),
#                     ybounds  = np.linspace( 0, 1, bounds[0] + 1 ),
                     colormap = chaco_colormaps.jet,
                     interpolation = 'nearest')

        rp.contour_plot( rates,
                         type = 'line',
#                         xbounds  = np.linspace( 0, 1, bounds[1] + 1 ),
#                         ybounds  = np.linspace( 0, 1, bounds[0] + 1 ),
                         bgcolor  = 'black',
                         levels   = 15,
                         styles   = 'solid',
                         widths   = list( np.linspace( 4.0, 0.1, 15 )),
                         colors   = gmt_drywet )

        rp.aspect_ratio = 1

#        zoom = ZoomTool( rp, tool_mode = 'box', always_on = False )
#        rp.overlays.append(zoom)

#        rp.tools.append( PanTool( rp, constrain_key = 'shift' ) )
        
        subplot = OverlayPlotContainer( )
        subplot.add( rp )
        
        return subplot


    def default_cmap( self, range, **traits ):

        """ Default colormap for brain slice plots """

        vrange = ( -256, 256 )

        _data = {'red':   ((0,    0,   0),
                           (0,    0,   0),
                           (0.5,  0.1, 0.1),
                           (0.85, 1.0, 1.0),
                           (1.0,  1.0, 1.0)),
                 'green': ((0,    1,    1),
                           (0.15, 0.5,  0.5),
                           (0.5,  0.,   0.),
                           (0.7,  0.,   0.),
                           (0.85, 0.85, 0.85),
                           (1.0,  1.0,  1.0)),
                 'blue':  ((0,    1,    1),
                           (0.40, 0.5,  0.5), 
                           (0.5,  0,    0),
                           (0.85, 0.,   0.),
                           (1.0,  1.,  1.)),
                 'alpha': ((0.,   1.,   1.), 
                           (0.4,  1.,   1.),
                           (0.495,0.,   0.),
                           (0.5,  0.,   0.),
                           (0.505,0.,   0.),
                           (0.6,  1.,  1.),
                           (1.0,  1.,  1.)) }

        cm = ColorMapper.from_segment_map( _data, range = range, **traits )        

        cm.low  = vrange[0]
        cm.high = vrange[1]

        return cm



class ParamController( MRIBrainContainer ):

#    coef_dict = coef_dict

    V_view = VGroup( 
        VGroup(
            Item( 'x_coord' ),
            Item( 'y_coord' ),
            Item( 'z_coord' ),
            Item( 't_coord' ),
            show_border = True,
            label       = 'Slice Coordinates'), 
        VGroup(
            Item( 'lambda_1' ),
            Item( 'lambda_2' ),
            Item( 'lambda_G' ),
            Item( 'delta' ),
            Item( 'lambda_1_star' ),
            Item( 'rate' ),
            show_border = True,
            label       = 'Model Parameters'),
        VGroup(
            Item( 'loss_type' ),
            Item( 'penalty_type' ),
            show_border = True,
            label       = 'Model Type'),
        VGroup( 
            Item( 'plot',
                  editor     = ComponentEditor(), 
                  show_label = False,
                  width      = 500,
                  height     = 500,
                  resizable  = True ),
            ),
        spring)
 

    H_view = HGroup( 
        VGroup(
            Item( 'x_coord' ),
            Item( 'y_coord' ),
            Item( 'z_coord' ),
            Item( 't_coord' ),
            show_border = True,
            label       = 'Slice Coordinates' ), 
        VGroup(
            Item( 'lambda_1' ),
            Item( 'lambda_2' ),
            Item( 'lambda_G' ),
            Item( 'delta' ),
            Item( 'lambda_1_star' ),
            show_border = True,
            label       = 'Model Parameters'),
        VGroup(
            Item( 'loss_type' ),
            Item( 'penalty_type' ),
            show_border = True,
            label       = 'Model Type' ),
        spring)

    
    traits_view = View( V_view,
                        title = 'Neuroparser',
                        kind = 'live'
                        )

    other_view = View( H_view,
                       title = 'Neuroparser',
                       kind = 'live'
                       )


    def __init__( self, **traits ):
        super( ParamController, self ).__init__( **traits )

        p = self.rate_plot ( "rates" ) 

        container = Plot()
        container.add( p ) 

        self.plot = container

    def rate_plot( self, rates, **traits ):

        rp = Plot( self.rate_data, default_origin = 'bottom left' )

        rp.x_axis.visible = False 
        rp.y_axis.visible = False 

        bounds = self.rate_data.get_data( rates ).shape

#        rp.img_plot( rates,
#                     xbounds  = np.linspace( 0, 1, bounds[1] + 1 ),
#                     ybounds  = np.linspace( 0, 1, bounds[0] + 1 ),
#                     colormap = chaco_colormaps.jet,
#                     interpolation = 'nearest')

        rp.contour_plot( rates,
                         type = 'line',
#                         xbounds  = np.linspace( 0, 1, bounds[1] + 1 ),
#                         ybounds  = np.linspace( 0, 1, bounds[0] + 1 ),
                         bgcolor  = 'black',
                         levels   = 15,
                         styles   = 'solid',
                         widths   = list( np.linspace( 4.0, 0.1, 15 )),
                         colors   = gmt_drywet )

        rp.aspect_ratio = 1

#        zoom = ZoomTool( rp, tool_mode = 'box', always_on = False )
#        rp.overlays.append(zoom)

#        rp.tools.append( PanTool( rp, constrain_key = 'shift' ) )
        
        subplot = OverlayPlotContainer( )
        subplot.add( rp )
        
        return subplot


def rate_contour_plot( rate_arr, rate_lut, idx ):

    # set matplotlib params
    from matplotlib import rcParams

    #set plot attributes
    fig_width  = 10  # width in inches
    fig_height = 10  # height in inches
    fig_size   = [fig_width,fig_height]
    params = {'backend': 'Agg',
              'axes.labelsize':  18,
              'axes.titlesize':  18,
              'font.size':       12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'figure.figsize':  fig_size,
              'savefig.dpi' :    600,
              'font.family':    'sans-serif',
              'axes.linewidth' : 0.5,
              'xtick.major.size' : 2,
              'ytick.major.size' : 2,
              }
    rcParams.update(params)
    import pylab as pl
    import matplotlib as mpl

    fig = pl.figure()
    ax  = fig.gca()
    X,Y = np.meshgrid( np.log(rate_lut['lG']+1), rate_lut['l1'])
    Z = rate_arr[:,idx,:]
    Z[ np.where(Z < 0.1) ] = 0.64
    Z[ np.where(Z > 1.) ]  = 0.64
    asp = len(rate_lut['lG'])/float(len(rate_lut['l1']))
    norm = mpl.colors.Normalize(vmin=0.64,vmax=0.735)

    im = pl.imshow(Z, aspect = asp,
                   interpolation = 'bilinear', 
                   cmap = my_cmap,
                   origin = 'lower',
                   norm   = norm )

    #set labels
    pl.xlabel(r'$\lambda_G$')
    pl.ylabel(r'$\lambda_1$')

    CB = pl.colorbar(im)
    pl.show()
    1/0

# Run if invoked from the command line:
def main():

#    MRIBrainContainer().edit_traits()
#    ParamController().configure_traits()
    rate_contour_plot( rate_array, rate_lut, 0)

if __name__ == '__main__':
    main() 



