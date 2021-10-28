

class cdl_acreage(object):
    def __init__(self, map_widget):
        self.m = map_widget
        
        
        ## need a simple draw control
        self.draw_control = ipyleaflet.DrawControl(
            edit=False,
            remove=False,
            circlemarker={},
            polyline={},
            polygon={},
            rectangle={
                "shapeOptions": {
                    "fillColor": "#d534eb",
                    "color": "#d534eb",
                    "fillOpacity": 0.2,
                }
            },
        )
        self.m.add_control(self.draw_control)
        ## When I draw something, do something
        ## 
        
        
        
    @property
    def df(self):
        if self._df is not None:
            return self._df
        else:
            raise RuntimeError("Must click on a point first")