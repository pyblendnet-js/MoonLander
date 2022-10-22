'''
    Renders a traingle that has all RGB combinations
    
    v2 = remove dependancies 
    v8 = auto cropping of enlarged moon
    v9 = initial ripple and rotation
    v10 = potato moon
    v11 = add object with alpha blend
    v12 = failed scaling of texture
    v13 = scaling success
    v14 = mouse sprite position without roll
    v15 = sprite rect render
    v17 = screenToSprite
    v26 = bonding almost working
    v29 = zoom in and out using <>
    v30 = add siper zoom on slash. added Moon over image
    v31 = flame following altitude of false horizon
    v32 = matching moon image with moon radius
    v33 = eagle rocket control
    v34 = added vehicle mass calcs - need to fault find
    v35 = we have movement
    v36 = working on origin offset
    v37 = origin offset good
    v38 = alt to zoom for divorced eagle.
    v39 = add gravity and counter force
    v40 = link rotation to speed
    v41 = add joystick control option
    v42 = add alt and fuel metres
    v43 = in progress
    v44 = button divorce, moon height, moon speed fix 
    v47 = eagle down, fireball
    v48 = add shadow (works well for close, but needs work for far) + fireball death
    v49 = more work on shadows and fireballs, added reset using other joystick buttons
    v50 = shadow work
    v51
    v52 = much better shadows
'''

import numpy as np
from pyrr import Vector3, Matrix44, matrix44

import moderngl
import moderngl_window as mglw
import math, random
from pathlib import Path

#for pygame
import pygame
from moderngl_window import geometry

import json
from os import path

from udp_server import udp_server_class

class Object2D:

    name = "Unknown"
    position = [0,0]  #can't use this initial value directly as it is a shared object
    angle = 0.0   #in degrees
    scale = 1.0
    origin = [0,0]  #shared values

    def __init__(self, ctx, aspect, name, texture, errors, **kwargs):
        super().__init__(**kwargs)
        
        self.name = name
        self.ctx = ctx
        self.aspect = aspect
        self.texture = texture
        self.errors = errors
        #rint("Texture:",dir(texture))
        #rint("Texture size:",texture.width, texture.height, texture.size)
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                
                in vec3 in_vert;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec2 v_text;

                void main() {
                    gl_Position =  Mvp * vec4(in_vert, 1.0);
                    v_vert = in_vert;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                uniform vec4 background;

                in vec3 v_vert;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    //f_color = texture(Texture, v_text).rgba;
                    vec4 tex = texture(Texture, v_text).rgba;
                    if(tex.w == 0) {
                      f_color = background;
                    } else { 
                      f_color = vec4(tex.x,tex.y,tex.z,tex.w);
                    }
                }
            ''',
        )
        w = texture.width
        h = texture.height
        verts = [
            # x, y, z,  text_x, text_y,
           -w, h, 0.0,    1.0, 0.0,
            w, h, 0.0,    0.0, 0.0, 
            w, -h, 0.0,   0.0, 1.0, 
           -w, -h, 0.0,   1.0, 1.0,   
        ]
        indis = [0, 1, 2, 2, 3, 0]
        vertices = np.array(verts,dtype='f4')
        #rint(vertices)
        indices = np.array(indis, dtype='i4')

        self.vbo = self.ctx.buffer(vertices)
        self.ibo = self.ctx.buffer(indices)

        self.mvp = self.prog['Mvp']
        self.background = self.prog['background']

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_vert to the first 3 floats
                # Map in_texcoord 2 floats
                (self.vbo, '3f 2f', 'in_vert', 'in_texcoord_0')
            ], self.ibo
        )
        
        self.position = [0,0]
      #endif
    #enddef

    def screenToSprite(self, sp):
      #rint("screen pos:",sp)
      #rint("pos:",self.position)
      px = sp[0]/self.scale/self.aspect - self.position[0]
      py = -self.position[1] - sp[1]/self.scale
      #rint("dp:",px,py)
      pv = pygame.Vector2(px,py)
      nv = pv.rotate(self.angle)
      return (nv.x,nv.y)
    #enddef  
    
    def render(self,proj,mode=5,ppos=None):
      self.texture.use()
      broll = Matrix44.from_eulers((0,math.radians(self.angle),0), dtype='f4')
      bscale = matrix44.create_from_scale((self.aspect*self.scale,self.scale,1.0), dtype='f4')
      bpos = Matrix44.from_translation((self.position[0]-self.origin[0],self.position[1]-self.origin[1],0), dtype='f4')
      #rint(type(ppos))
      if type(ppos) == type(None):
        btot = proj*bscale*bpos*broll  #generally this means not a child
      else:
        btot = proj*bscale*broll*ppos  #generally a child offset
      #endif
      self.mvp.write(btot)
      self.vao.render(mode)  #mode 5 = triangle strip
      return btot
    #enddef 
    
    def logError(self,err):
      if err in self.errors:
        self.errors[err] += 1
      else:
        self.errors[err] = 1
      #endif
    #enddef         
   
#endclass

class Lines():
    def __init__(self, ctx, aspect, verts, obj=None, **kwargs):       
        self.ctx = ctx
        self.aspect = aspect
        self.obj = obj

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec2 in_vert;

                //out vec3 v_vert;

                void main() {
                    gl_Position =  Mvp * vec4(in_vert, 0.0, 1.0);
                    //gl_Position =  vec4(in_vert, 0.0, 1.0);
                    //v_vert = in_vert;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec4 v_color;
                out vec4 f_color;

                void main() {
                  f_color = v_color;
                }
            ''',
        )
        #verts = [0.5,0, -0.5,0,
        #         0,0.5, 0,-0.5]
        vertices = np.array(verts,dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.mvp = self.prog['Mvp']
        self.color = self.prog['v_color']
        self.color.value = (1.0,0.7,0.7,1.0)
        self.vao = self.ctx.vertex_array(self.prog,[(self.vbo, '2f', 'in_vert')])
    #enddef 

    def render(self,proj,time):
        if int(time*2) & 1 == 0: 
          self.color.value = (1.0,0.7,0.7,1.0)
        else:  
          self.color.value = (0.0,0.0,0.0,1.0)
        #endif
        if self.obj:
          #rint(self.obj.position)
          '''broll = Matrix44.from_eulers((0,self.obj.angle,0), dtype='f4')
          bscale = matrix44.create_from_scale((self.aspect*self.obj.scale,self.obj.scale,1.0), dtype='f4')
          bpos = Matrix44.from_translation((self.obj.position[0],self.obj.position[1],0), dtype='f4')
          btot = proj*bscale*bpos*broll'''
          self.mvp.write(proj)
        else:
          bscale = matrix44.create_from_scale((self.aspect,1.0,1.0), dtype='f4')
          self.mvp.write(bscale)
        #endif    
        self.vao.render(moderngl.LINES)
    #enddef
#endclass    

class Sprite(Object2D):
    lines = None
    visible = True
    saved = True
    selected = False
    typ = "Unknown"
    children = []  #{sprite:,anchor:}
    bond = None  #defaults to centre
    parent = None
    divorced = False
    predivorceState = None
    rot_vel = 0
    down = False
        
    def __init__(self,asset_path="",family=None,**kwargs):
      super().__init__(**kwargs)
      self.asset_path = asset_path
      self.family = family #dictionary to shared sprite container
      #rint("before:",self.locs)
      self.locs = {"centre":(0,0)} #dict of points within object
      self.load()
      #rint(self.name,"after:",self.locs)
      self.acc = [0,0]
      self.vel = [0,0]
    #enddef
    
    def load(self):
      fid = self.asset_path + self.name + ".json"
      try:
        f = open(fid)
      except:
        print("No config for sprite " + fid)
        return
      #endif    
      config = json.load(f)
      #rint("config:",config)
      self.setState(config)
    #enddef
    
    def save(self):
      config = { "locs":self.locs }
      with open(self.asset_path + self.name + ".json", 'w') as f:
        json.dump(config, f)
      self.saved = True
    #enddef
    
    def setLoc(self,pos):
      #find unique name
      li = 0
      loc_nm = ""
      #rint("Locs:",self.locs)
      while True:
        loc_nm = "loc" + str(li)
        #rint("Test:",loc_nm)
        if not loc_nm in self.locs.keys():
          break
        #endif
        li += 1
      #endwhile    
      self.locs[loc_nm] = pos
    #enddef
    
    def getStatus(self):
      rtn =  "[%0.1f:%0.1f]"%tuple(self.position) + " A%d"%(self.angle,)
      if self.parent:
        rtn += " P:" + self.parent.name
        if self.divorced:
          rtn += "D S%0.3f"%(self.scale)
        #endif
      else:
        rtn += " S%0.3f"%(self.scale)
      #endif
      return rtn   
    #enddef
      
    def getState(self):
      return {"name":self.name,"typ":self.typ,"position":list(self.position),"angle":self.angle,"scale":self.scale,"children":self.children,"bond":self.bond, "divorced":self.divorced}
    #endif
    
    def setState(self,state):
      for k in state.keys():
        if hasattr(self,k):
          setattr(self,k,state[k])
          #rint("Set:",k," to ",state[k])
        #endif
      #endfor    
    #enddef

    def nearest(self, pos, dist = 1000):
      p = pygame.Vector2(pos)
      best = None
      for lk in self.locs.keys():
        loc = self.locs[lk]
        l = pygame.Vector2(loc)
        d = p.distance_to(l)
        if d < dist:
          dist = d
          best = lk
        #endif
      #endfor
      return dist, best
    #enddef
        
    def showLocs(self,state = True):
      if state:
        s = max(self.texture.size)*0.05
        verts = []
        for loc in self.locs.values():
          verts += [loc[0]-s,loc[1],loc[0]+s,loc[1],loc[0],loc[1]-s,loc[0],loc[1]+s]
        #endfor   
        self.lines = Lines(ctx=self.ctx,aspect=self.aspect,verts=verts,obj=self)
      else:
        self.lines = None  
    #enddef
    
    def showBackground(self,state = True):
      if state:
        self.background.value = (0.5,0.0,0.0,1.0)
      else:
        self.background.value = (0,0,0,0)
      #endif
    #enddef
    
    def divorce(self,aspect):
      if not self.parent:
        return
      #endif  
      if self.divorced: #already divorced
        return
      #endif 
      self.vel = list(self.parent.vel)
      self.rot_vel = self.parent.rot_vel
      self.predivorceState = self.getState()
      sp = self
      r = 0
      s = 1
      dp = pygame.Vector2(0,0)
      print("Divorce:",sp.name)
      while True:
        r += sp.angle 
        #rint("pos:",dp)
        if sp.bond:  #this is point on the child where the parent is attached
          try:
            bond = sp.locs[sp.bond]
            #rint("Bond point:",bond)
            dp -= pygame.Vector2(bond)
          except:
            self.logError("No anchor:" + sp.bond + " in sprite:" + sp.name)
          #endtry  
        #endif
        dp = dp.rotate(-sp.angle)  #adjust for rotation of child
        dp += pygame.Vector2(sp.position) #position not effected by rotation, only scale
        #rint("pos2:",dp)
        s *= sp.scale
        #rint("Mod loop",sp.name,sp.parent.name) 
        if not sp.parent or sp.divorced:
          break
        #endif
        #rint("Parent:",sp.parent.name," Angle:",r," Scale:",s)
        for child in sp.parent.children:
          if child["sprite"] == sp.name:
            #rint("Found reference to:",sp.name)
            try:
              loc = sp.parent.locs[child["anchor"]]  #this is point on the parent where the child is attached
              #rint("loc:",loc," anchor:",child["anchor"])
              dp += pygame.Vector2(loc)
              break
            except:
              self.logError("No anchor:" + child["anchor"] + " in sprite:" + sp.parent.name)
            #endtry
          #endif
        #endfor 
        sp = sp.parent
        #rint("End of loop",sp.name,sp.parent.name) 
      #endwhile
      self.angle = r
      self.scale = s
      self.position[0] = dp[0]
      self.position[1] = dp[1]
      print(self.name," Angle:",r," Scale:",s," Pos:",self.position)
      self.aspect = aspect  #now that it is as screen level, it needs to apply aspect for render
      #rint("DD:",self.divorced)
      self.divorced = True  #must happen after getState
    #enddef
    
    def undivorce(self):
      if not self.divorced: #already has parent
        return
      #endif  
      self.divorced = False
      if not self.parent:
        return;
      #endif  
      #rint(self.predivorceState)
      if self.predivorceState:
        self.setState(self.predivorceState)
      else:  #otherwise do it the hard way
        p = self.parent
        ancestors = []
        while p:
          ancestors.insert(0,(p,self))
          p = p.parent
        #endwhile
        r = 0.0
        for p, s in ancestors: 
          if not s.bond: #child has no bond point, so determine the correct position offset      
            dx = 0
            dy = 0
            for child in p.children:
              if child["sprite"] == s.name:
                loc = p.locs[child["anchor"]]
                dx += loc[0]
                dy += loc[1]
                #rint("anchor:",dx,dy)
                break
              #endif
            #endfor 
            #rint("Scale:",self.scale, p.scale)   
            pv = pygame.Vector2(dx,dy)
            #rint(pv)
            nv = pv.rotate(-p.angle)
            #rint("pos:",self.position," nv:",nv)
            #if s.bond:
            #bond = s.locs[self.bond]
            #bv = pygame.Vector2(bond)
            #mv = bv.rotate_rad(-r)
            #nv -= mv
            #endif
            #rint("bond:",nv)
            self.position[0] += nv[0]*s.scale     
            self.position[1] += nv[1]*s.scale
            print(self.name," pos:",self.position," nv:",nv)
            exit(1)
          #endif  
          self.angle -= p.angle
          self.scale /= p.scale
          r = p.angle
          p = p.parent  #go to next level up
          #rint(self.getState())
        #endwhile
        if self.bond:  #child is bonded to parent, so no need for pos offset
          self.position = [0,0]  
        #end
      #endif    
      self.aspect = 1.0
    #enddef
        
    def render(self,proj,time,mode=5,ppos=None):
      if not self.visible:
        return
      #endif  
      btot = super().render(proj,mode,ppos)
      if self.lines:
        self.lines.render(btot,time)
      #endfor
      #rint("family:",self.family)
      for child in self.children:
        #rint("child:",child)
        try:
          sprite = self.family[child["sprite"]]
        except:
          continue
        #endtry  
        if sprite.divorced:
          continue
        #endif
        sprite.origin = self.origin  #use parent's origin
        try:
          anchor_pos = self.locs[child["anchor"]]
        except:
          print("No anchor:" + child["anchor"] + " in child sprite:" + self.name)
          #rint(self.locs)
          exit(1)
          self.logError("No anchor:" + child["anchor"] + " in child sprite:" + self.name)
          sprite.divorce(self.aspect)  #temporary so anchor can be fixed
          return
        #endtry    
        #rint("Name:",self.name," Child:",sprite.name," Bond:",sprite.bond)
        if sprite.bond != None:
          try:
            child_anchor = sprite.locs[sprite.bond]
            #rint("Child anchor:",child_anchor,anchor_pos)  #sprite.bond,sprite.locs)
          except:
            print("No anchor:" + sprite.bond + " in parent sprite:" + sprite.name)
            #rint(sprite.locs)
            exit(1)
            self.logError("No anchor for " + sprite.bond + " in parent sprite:" + sprite.name)
            ppos = None  
          #endtry
          ppos = Matrix44.from_translation((-child_anchor[0],-child_anchor[1],0), dtype='f4')
        else:
          ppos = None
        #endif
        bpos = Matrix44.from_translation((anchor_pos[0],anchor_pos[1],0), dtype='f4')
        #endif
        sprite.render(btot*bpos,time,mode,ppos)
      #endfor  
    #enddef    
#enddef        


# moon is 1737km radius, gravity 1.62m/s2, -4 to +7km altitiude
# screen radius is 1, so scale 1 screen unit = 1737000m
# command module orbited at ave 104km, 1633m/sec, 7085sec/rev  (Apollo14 211km, 120min=7200sec)
# 11m long by 4m diam; 80pix = 4m => 20 screen units/metre by image @ scale 1
# 0.02 per metre at 0.001scale, or 1 = 50m?
# difference between the two is 20 x 1737000 = 34740000 => 0.000000029
# if sprites scales are 0.001, then moon scale is 34740

class MoonObject():
    segs = 360
    moon_pos = [0.0,0.0]
    moon_scale = 1.0
    moon_angle = 0.0
    aspect = 0.6
    pi = 3.1415927
    pi2 = pi/2.0
    moon_angle_start = 0.0
    moon_angle_inc = pi2/30
    
    def __init__(self, ctx, aspect, **kwargs):
        super().__init__(**kwargs)
        
        self.ctx = ctx
        self.aspect = aspect

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                
                uniform vec3 angle;
                uniform vec3 scale;
                uniform vec3 pos;
                uniform float terr[20];
                uniform vec3 centre_color;
                uniform vec3 edge_color;
                uniform float radius;

                in float in_i;

                out vec3 v_color;    // Goes to the fragment shader

                void main() {
                    float x = pos[0];
                    float y = pos[1];
                    int ai = int(in_i)/3;
                    int i = int(in_i) % 3;
                    if(i == 0) {
                      v_color = centre_color;
                    } else {
                      if(i == 2)  // 3rd vertex in triangle is one step along
                        ai++;
                      float a = angle[1] + ai*angle[2];
                      float r = radius;
                      for(int i=0; i < 20; i+= 2)
                        r += terr[i]*(0.2 + sin((a + angle[0])*terr[i+1]));
                      x -= sin(a)*r*scale[0];
                      y -= cos(a)*r*scale[1];
                      v_color = edge_color*(1.0-cos(a));
                    }  
                    gl_Position = vec4(x, y, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                in vec3 v_color;
                out vec4 f_color;

                void main() {
                    // We're not interested in changing the alpha value
                    f_color = vec4(v_color, 1.0);
                }
            ''',
        )
        self.angle = self.prog['angle']
        self.scale = self.prog['scale']
        self.pos = self.prog['pos']
        self.pos.value = (self.moon_pos[0],self.moon_pos[1],0.0)
        self.adj_scale(1.0)
        self.terr = self.prog['terr']
        self.radius = self.prog['radius']
        self.radius.value = 0.990  #0.986
        centre_color = self.prog['centre_color']
        centre_color.value = (0,0,0)
        edge_color = self.prog['edge_color']
        edge_color.value = (0.2,0.2,0.2)
        print("Terr Array length:",self.terr.array_length)
        terr = []
        h = 0.001
        d = 20
        for i in range(0,self.terr.array_length,2):
          terr += [h,d]
          h *= 0.6
          d *= 2
        #endfor  
        self.terr.value = terr
        self.terr_shadow = terr
        verts = []
        #for i in range(360):
        #  verts += [i,]  # i
        verts = list(range(self.segs*3))   #360 for 3 degree increments
        #endfor  
        vertices = np.array(verts,dtype='f4')
        self.vbo = self.ctx.buffer(vertices)

        # We control the 'in_vert' and `in_color' variables
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_i float
                (self.vbo, 'f', 'in_i')
            ],
        )
    #enddef    

    def render(self):
        #unless all of the moon is visible, then not all of the moon is rendered
        # moon_angle = current rotation
        # moon_angle_start = start of arc from left hand side
        # segment size
        self.angle.value = (self.moon_angle, self.moon_angle_start, self.moon_angle_inc)
        self.vao.render()
    #enddef
    
    def get_alt(self, ai = 180):  #assume segments = 360
        a = self.moon_angle_start + ai*self.moon_angle_inc
        r = self.radius.value
        for i in range(0,20,2):
          r += self.terr_shadow[i]*(0.2 + math.sin((a + self.moon_angle)*self.terr_shadow[i+1]))
        #endfor
        return r # where 1.0 is the circumfrence of moon
    #enddef     
        
    def get_point(self,a):
        x = self.moon_pos[0] - math.sin(a)*self.moon_scale
        y = self.moon_pos[1] - math.cos(a)*self.moon_scale
        return(x,y)
    #enddef
        
    def adj_angle(self):
        if self.moon_pos[1] < -1.0:
          '''#rint("Reduce moon displayed")
          #self.angle.value = (self.moon_angle, self.pi2, self.pi2/60)
          print("Start pos:",self.get_point(self.pi2))
          print("End   pos:",self.get_point(self.pi2*3))
          lx = 1.2/self.aspect    
          sx = ((self.moon_pos[0] + lx)/self.moon_scale)
          if sx < 1.0 and sx > -1.0:
            ta = math.asin(sx)         
            self.moon_angle_start = self.pi - ta
            #auto lover the moon
            #self.moon_pos[1] = -1.0-(math.sqrt(1 - sx*sx))*self.moon_scale
            #self.pos.value = (self.moon_pos[0],self.moon_pos[1],0.0) 
          else:'''
          ta = math.atan(1/(self.moon_pos[1]+1.0)/self.aspect)
          #print(self.moon_pos[1]+1.0,math.degrees(ta)) 
          self.moon_angle_start = self.pi - ta
          #endif  
          '''ex = ((self.moon_pos[0] - lx)/self.moon_scale)
          if ex < 1.0 and ex > -1.0:
            ta += math.asin(-ex)
          else:'''  
          #ta = -self.pi/2
          #endif  
          self.moon_angle_inc = 2.0*ta/self.segs
          #rint("Arc:",math.degrees(self.moon_angle_start - self.pi)*2," Inc:",math.degrees(self.moon_angle_inc))
        else:
          self.moon_angle_start = 0.0
          self.moon_angle_inc = self.pi*2.0/self.segs
        #endif
    #enddef
        
    def adj_scale(self, adj):
        self.moon_scale *= adj
        self.scale.value = (self.moon_scale*self.aspect,self.moon_scale,1.0)
        self.adj_angle()
    #enddef
    
    def adj_pos(self,dx,dy): 
        self.moon_pos[1] += dy
        self.pos.value = (self.moon_pos[0],self.moon_pos[1],0.0) 
        self.adj_angle()               
    #enddef
    
    def set_pos(self):
        self.pos.value = (self.moon_pos[0],self.moon_pos[1],0.0) 
    #enddef
    
#endclass

def interp(f,s,e):
  return s + (e-s)*f
#enddef  

class game(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1280, 720)
    aspect_ratio = None #window_size[0] / window_size[1]
    resizable = True
    title = "Moon Lander"
    pause = False
    resource_dir = (Path(__file__)/ '..').absolute()
    
    arrow_key_mode = 0
    edit_object = None
    
    last_gui_refresh = 0  #used for reducing gui refresh
    render_cnt = 0   #used to calc FPS
    
    sprites = {}
    showlocs = False
    showhidden = False
    origin_pos = [0,0] # can be set to a sprite pos to centre object
    
    shift_pressed = False
    cntrl_pressed = False
    alt_pressed = False
    gui = {}
    
    sprite_win = None
    errors = {}
    
    modeChange = 2
    modeFrac = 0
    currentMode = 0
    
    rocket_left_dn = 0
    rocket_command = 0
    rocket_right_dn = 0
    rocket_left = 0
    rocket_eagle = 0
    rocket_right = 0
    rocket_left_up = 0
    rocket_descent = 0
    rocket_right_up = 0
    
    #ref: https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1969-059C
    
    atti_fuel = 287 #kg
    atti_init_fuel = 287 #kg
    atti_thrust = 440*2  #normally work in quads
    atti_flow = 5 #kg/sec is big guess
    atti_dx = 2
    descent_init_fuel = 8248
    descent_fuel = descent_init_fuel
    descent_thrust = 45000  #throttleable from 10 to 60%
    descent_mass = 2134 #tot 10334 from https://en.wikipedia.org/wiki/Apollo_Lunar_Module)
    descent_rot_mass = 2034*4
    descent_burn_tm = 13*60   # 311sec*9.8*flow_rate = 45000 => flow_rate = 14.7kg/sec => 561sec  (13min=780sec)
    descent_flow = 140  # acutally 14   
    eagle_init_fuel = 2590  #4740 - 2150 =     # from https://en.wikipedia.org/wiki/Apollo_Lunar_Module
    eagle_fuel = eagle_init_fuel
    eagle_thrust = 15000
    eagle_mass = 2150   #kg
    eagle_rot_mass = 1000   #guess based on videos
    #  15000N/9.8/311= 4.9kg/sec => 484sec
    eagle_flow = 50  # actually 5
    eagle_burn_tm = 484   #calculated
    command_fuel = 28800-11900
    command_init_fuel = command_fuel
    command_burn_tm = 750
    command_thrust = 92000  #N
    command_mass = 11900  #kg dry
    command_rot_mass = 11900*5
    command_flow = 30
    # 92000/9.8/314=30kg/sec=>563sec
    pos_scale = 20
    screenToKm = 0.05  #screen units to km  (so screen height = 100m)
    kmToScreen = 1/screenToKm # m to sprite units (remember sprite scale = 0.001)
    spriteToKm = screenToKm*0.001
    kmToSprite = 1/spriteToKm
    minEagleHeight = 15000
    orbit_height = 0.2
    max_orbit_height = 2
    proper_orbit_height = 104  #km
    alt = orbit_height
    orbit_speed = 150   #real speed = 1515  #m/s or 1633 m/s? Reduced so that it doesn't take 500seconds to deorbit
    # total eagle mass for descent = 4740 + 10334 = 15074kg =>45000/15000 = 3m/s2 => 1515/3 = 500sec (must be less due to lower weight after fuel is used)
    moon_gravity = 1.625 #m/s2
    #moon_diam = 3472km
    #circumfrence = 10908 km => 5454km/hr => 1515m/s
    
    joystick_zero = [0,0]
    joystick_server = None
    jbtns = [0]*12
    jx = 0
    jy = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aspect = self.window_size[1]/self.window_size[0]
        self.moon = MoonObject(self.ctx,self.aspect)
        self.createOffScreenSurface()
        self.sprite_win = pygame.Rect(10,10,200,self.window_size[1]-20)
        self.reset()
        #self.loadConfig()
        pygame.joystick.init()
        self.joysticks = pygame.joystick.get_count()
        self.joyimg = pygame.image.load("images/attack3.png")
        print("Joysticks:",self.joysticks)
        if self.joysticks > 0:
          self.joystick_server = udp_server_class()
          #pygame joystick doesn't work well with modernGl_window
          #joystick.init()
          #print("Init joystick ",joystick.get_numbuttons()," buttons,",joystick.get_numaxes()," axis")
        #endif
        #self.joystick_server = udp_server_class() 
    #enddef

    def loadSprites(self, sprites):
      for nm in sprites:
        state = sprites[nm]
        #rint("Nm:",nm," State:",state)
        #rint(state["typ"])
        tex = self.load_texture_2d("apollo/" + state["typ"] + ".png")
        rep = 1  #first instance
        unm = nm
        while unm in self.sprites:
          unm = nm + str(nm)
          rep += 1
        #endwhile  
        sprite = Sprite(ctx=self.ctx,aspect=self.aspect,name=unm,texture=tex,asset_path="apollo/",family=self.sprites, errors = self.errors)
        fid = "apollo/" + state["typ"] + ".locs"
        #rint("Alt loc file:",fid)
        if path.exists(fid):
          try:
            f = open(fid)
          except:
            print("No locs for " + fid)
            return
          #endif    
          if sprite.locs:
            #rint(sprite.locs)
            #sprite.logError("Adding locs from "+fid+" to "+sprite.name)
            locs = json.load(f)
            for l in locs.keys():
              #rint(l)
              sprite.locs[l] = locs[l]
            #endfor
            #rint(sprite.locs) 
          else:
            #sprite.logError("Using locs from "+fid," to "+sprite.name)
            sprite.locs = json.load(f)
          #endif  
        #endif 
        self.sprites[unm] = sprite
        sprite.setState(state)
      #endfor
      #register which sprites are currently children
      for sprite in self.sprites.values():
        for child in sprite.children:
          try:
            child_sprite = self.sprites[child["sprite"]]
          except:
            print(self.sprites)
            print("No ",child["sprite"]," found")
            exit(0)  
          if child_sprite.parent != None:
            print("Sprite:" + child_sprite.name + " has more than one parent")
          else:
            child_sprite.parent = sprite
            #rint("Child:",child_sprite.name," has parent:",sprite.name)
            if not child_sprite.divorced:
              child_sprite.aspect = 1.0  #aspect only needs to be applied once
            #endif  
          #endif  
        #endfor
      #endfor
    #enddef
    
    def reset(self):
      self.sprites = {}
      self.loadConfig()
      command = self.sprites["command"]
      eagle = self.sprites["eagle"]
      descent = self.sprites["descent"]
      eagle.down = False
      descent.down = False
      self.eagle_fuel = self.eagle_init_fuel
      self.descent_fuel = self.descent_init_fuel
      self.command_fuel = self.command_init_fuel
      self.atti_fuel = self.atti_init_fuel
      self.currentMode = 0
      self.modeChange = 2
      self.modeFrac = 0
      self.showCrashBlurb = False
      self.showSuccessBlurb = False
      self.showWait = True
      self.showInstruction = False
      self.sprites["shadow"].visible = False
    #enddef   
    
    def setState(self,state):
      for k in state.keys():
        if hasattr(self,k):
          setattr(self,k,state[k])
        #endif
      #endfor    
    #enddef
   
    def loadConfig(self):
      fid = "game.json"
      try:
        f = open(fid)
      except:
        print("No config for " + fid)
        return
      #endif    
      config = json.load(f)
      #rint("config:",config)
      self.setState(config)
      self.loadSprites(config["actors"])
    #enddef
    
    def saveConfig(self):
      actors = {}
      for sp in self.sprites:
        sprite = self.sprites[sp]
        actors[sp] = sprite.getState()
      #endfor 
      config = { "actors":actors }
      with open("game.json", 'w') as f:
        json.dump(config, f)
      #endwith
    #enddef
    
    def readJoystick(self,time=None):
        if not self.joystick_server:
          #print("No")
          return -1
        #endif  
        rocket = 0
        while True:
          message,address = self.joystick_server.get()
          if address:
            msg = message.decode("utf-8")
            if msg[0] in "ud":
              bp = msg[2:].split(",")
              self.jbtns = []
              for b in bp:
                self.jbtns.append((b == '1'))
              #endfor  
              #rint("Buttons:",self.jbtns)
            else:
              ap = msg.split(",")
              self.jx = float(ap[0]) - self.joystick_zero[0]
              self.jy = float(ap[1]) - self.joystick_zero[1]
              if self.joystick_zero[0] == 0:
                self.joystick_zero[0] = self.jx
                self.joystick_zero[1] = self.jy
                self.jx = 0
                self.jy = 0
              #endif    
            #endif  
          else:
            break
          #endif
        #endwhile  
        #print("%10.3f Joystick %4.2f %4.2f"%(time,jx,jy),trigger,trigs)
        self.rocket_left_up = 0
        self.rocket_left_dn = 0
        self.rocket_right_up = 0
        self.rocket_right_dn = 0
        self.rocket_eagle = 0  
        self.rocket_descent = 0  
        if self.jx < -0.1:
          self.rocket_left_up = -self.jx
          self.rocket_right_dn = -self.jx
        elif self.jx > 0.1:
          self.rocket_left_dn = self.jx
          self.rocket_right_up = self.jx
        #endif
        if not self.jbtns[0]:
          if self.jy < -0.1:
            self.rocket_left_dn = -self.jy
            self.rocket_right_dn = -self.jy
          elif self.jy > 0.1:
            self.rocket_left_up = self.jy
            self.rocket_right_up = self.jy
          #endif
        else:
          rocket = (1.0-self.jy)/2
        #endif
        return rocket
    #enddef

    def render(self, time: float, frame_time: float):
        #first determine if there is view mode change in effect
        if self.modeChange > 0:
          #ease accel at start and finish
          if self.modeFrac < 0.5:
            mr = self.modeFrac + 0.001
          else:
            mr = 1.0 - self.modeFrac + 0.001
          #endif
          if not self.pause:  
            self.modeFrac += frame_time*mr
          #endif  
          if self.modeFrac > 1.0:  #finish mode change
            self.modeFrac = 1.0
            self.adjScalePos(self.currentMode,self.modeChange-1)
            self.currentMode = self.modeChange-1
            self.modeChange = 0
            self.modeFrac = 0
            self.showWait = False
            self.showInstructions = True
          else:
            self.adjScalePos(self.currentMode,self.modeChange-1)
          #endif
          #rint("modeFrac:",self.modeFrac)
        #endif    
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        #self.moon.moon_angle += frame_time / self.moon.moon_scale
        self.sprites["moon"].angle = -math.degrees(self.moon.moon_angle) 
        self.moon.render() 
        #if self.origin_pos:
        #  pos = Vector3(self.origin_pos)*-1 
        #else:
        pos = Vector3([0,0,0])
        #endif
        translation = Matrix44.from_translation(pos, dtype='f4')
        for sprite in self.sprites.values():
          if sprite.parent != None and not sprite.divorced:
            continue
          #endif
          if not sprite.name in ("moon","flame","altimeter"):
            sprite.origin = self.origin_pos
          #endif
          sprite.render(translation, time)
        #endif  
        self.pg_texture.use()
        self.render_cnt += 1
        if time > (self.last_gui_refresh+0.2):  #200mS refresh
          fps = self.render_cnt/(time - self.last_gui_refresh)
          self.render_pygame(time,fps)
          self.last_gui_refresh = time
          self.render_cnt = 0
        #endif  
        self.quad_fs.render(self.texture_program)     
        self.ctx.disable(moderngl.BLEND)
        land_pos = (self.moon.get_alt()-self.moon.radius.value)*self.moon.moon_scale*1000-500
        self.sprites['flame'].position =(0,land_pos)  #sprite scale is 0.001
        
        if self.pause or self.modeChange != 0:
          return
        #endif  

        command = self.sprites["command"]
        eagle = self.sprites["eagle"]
        descent = self.sprites["descent"]
        shadow = self.sprites["shadow"]
        rocket = self.readJoystick(time)
        if rocket >= 0:  #valid joystick
          print("Rocket:",rocket)   
          if self.jbtns[5] or self.jbtns[6] or self.jbtns[7] or self.jbtns[8] or self.jbtns[9] or  self.jbtns[10] or self.jbtns[11]:
            self.reset()     #won't use self.jbtns[3] or self.jbtns[4] for this
          elif eagle.divorced:
            if descent.divorced:
              self.rocket_eagle = rocket   #apply rocket to eagle ascent rocket
            else:
              if self.jbtns[2]:  #undock from descent lander
                descent.divorce(self.aspect)
                eagle.down = False
                self.showSuccess = False
              else:      
                self.rocket_descent = rocket  #otherwise apply rocket throttle to descent rocket
              #endif  
            #endif
          else:
            if self.jbtns[1]:  #undock from command module    
              eagle.divorce(self.aspect)
              self.sprites["shadow"].visible = True
            else:
              self.rocket_command = rocket  #otherwise apply rocket throttle to command module rocket
            #endif
          #endif  
          if self.jbtns[3]:
            self.rocket_left = 1
          else:
            self.rocket_left = 0  
          if self.jbtns[4]:
            self.rocket_right = 1
          else:
            self.rocket_right = 0      
          #endif      
        #endif      
        
        mass = self.eagle_mass + self.eagle_fuel + self.atti_fuel
        rot_mass = self.eagle_rot_mass
        centre = 0  #assumes centre of mass in line with thruster
        vehicle = eagle
        if not eagle.divorced:
          mass += self.command_mass + self.command_fuel
          rot_mass += self.command_rot_mass
          centre = 5  #towards command module
          vehicle = command
        #endif
      
        fx = 0
        fy = 0
        fr = 0
        #adjust origin to match object vehicle, also move neutral down as approaching ground
        if self.alt > 0.02:  # and self.modeFrac < 1.0:
          top_high = vehicle.position[1] - self.modeFrac*500 #screen to stop scrolling after 1/2 screen
          #rint("Top high:",top_high)
          self.origin_pos[1] = top_high
        #enddef
        self.origin_pos[0] = self.origin_pos[0]*(1-frame_time*10) + vehicle.position[0]*frame_time*10
        #rint("Origin:",self.origin_pos,vehicle.position[0])
        if not descent.divorced:
          mass += self.descent_mass + self.descent_fuel
          rot_mass += self.descent_rot_mass
          centre -= 1  #towards descent
        #endif
        eagle.acc = [0,0]
        command.acc = [0,0]
        if self.rocket_left != 0:
          fx -= self.atti_thrust*self.rocket_left
          self.atti_fuel -= self.atti_flow*frame_time*self.rocket_left
          self.sprites["eagle_left_thruster"].visible = True
          #self.sprites["eagle_left_thruster"].scale = self.rocket_left
        else:  
          self.sprites["eagle_left_thruster"].visible = False
        #endif
        if self.rocket_right != 0:
          fx += self.atti_thrust*self.rocket_right
          self.atti_fuel -= self.atti_flow*frame_time*self.rocket_right
          self.sprites["eagle_right_thruster"].visible = True
        else:  
          self.sprites["eagle_right_thruster"].visible = False
        #endif    
        if self.rocket_left_up != 0:
          fy -= self.atti_thrust*self.rocket_left_up
          fr += self.atti_thrust*self.atti_dx*self.rocket_left_up
          self.atti_fuel -= self.atti_flow*frame_time*self.rocket_left_up
          self.sprites["eagle_left_up_thruster"].visible = True
          self.sprites["eagle_left_up_thruster"].scale = self.rocket_left_up*0.5
        else:  
          self.sprites["eagle_left_up_thruster"].visible = False
        #endif
        if self.rocket_right_up != 0:
          fy -= self.atti_thrust*self.rocket_right_up
          fr -= self.atti_thrust*self.atti_dx*self.rocket_right_up
          self.atti_fuel -= self.atti_flow*frame_time*self.rocket_right_up
          self.sprites["eagle_right_up_thruster"].visible = True
          self.sprites["eagle_right_up_thruster"].scale = self.rocket_right_up*0.5
        else:  
          self.sprites["eagle_right_up_thruster"].visible = False
        #endif    
        if self.rocket_left_dn != 0:
          fy += self.atti_thrust*self.rocket_left_dn
          fr -= self.atti_thrust*self.atti_dx*self.rocket_left_dn
          self.atti_fuel -= self.atti_flow*frame_time*self.rocket_left_dn
          self.sprites["eagle_left_dn_thruster"].visible = True
          self.sprites["eagle_left_dn_thruster"].scale = self.rocket_left_dn*0.5
        else:
          self.sprites["eagle_left_dn_thruster"].visible = False
        #endif
        if self.rocket_right_dn != 0:
          fy += self.atti_thrust*self.rocket_right_dn
          fr += self.atti_thrust*self.atti_dx*self.rocket_right_dn
          self.atti_fuel -= self.atti_flow*frame_time*self.rocket_right_dn
          self.sprites["eagle_right_dn_thruster"].visible = True
          self.sprites["eagle_right_dn_thruster"].scale = self.rocket_right_up*0.5
        else:  
          self.sprites["eagle_right_dn_thruster"].visible = False
        #endif    
        if self.rocket_descent != 0:
          fy += self.descent_thrust*self.rocket_descent
          self.descent_fuel -= self.descent_flow*frame_time*self.rocket_descent
          self.sprites["descent_rocket"].visible = True
          self.sprites["descent_rocket"].scale = self.rocket_descent*2
        else:  
          self.sprites["descent_rocket"].visible = False
        #endif    
        if self.rocket_eagle != 0:
          fy += self.eagle_thrust*self.rocket_eagle
          self.eagle_fuel -= self.eagle_flow*frame_time*self.rocket_eagle
          self.sprites["eagle_rocket"].visible = True
          self.sprites["eagle_rocket"].scale = self.rocket_eagle*2
        else:
          self.sprites["eagle_rocket"].visible = False
        #endif    
        if self.rocket_command != 0:
          if eagle.divorced:
            cfy = -self.command_thrust*self.rocket_command
            command.acc[0] = cfy/(self.command_mass+self.command_fuel)
            #rint(command.acc[0])
          else:
            fy -= self.command_thrust*self.rocket_command
            #rint(fy)
          #endif    
          self.command_fuel -= self.command_flow*frame_time*self.rocket_command
          self.sprites["command_rocket"].visible = True
          self.sprites["command_rocket"].scale = self.rocket_command*4
        else:
          self.sprites["command_rocket"].visible = False
        #endif
        vehicle.acc[1] = fx/mass
        fr -= fx*centre  #added rotation   
        vehicle.acc[0] = fy/mass
        vehicle.rot_vel += fr/rot_mass*frame_time
        lp = land_pos + self.origin_pos[1]
        shadow.position[0] = eagle.position[0]
        if eagle.divorced:  #not connected so can fly by itself
          self.showInstructions = False
          self.service(command,frame_time)
          if eagle.down:
            self.alt = 0
            self.serviceCrash("eagle_fireball",eagle,land_pos,frame_time)
            #rint("Rot vel:",eagle.rot_vel,"Down vel:",eagle.vel[1])
            '''if eagle.rot_vel == 0 and eagle.vel[1] == 0:
              self.reset()
              print("got to herE")
              exit(1)
              return
            #endif '''
          else:
            km = eagle.position[1]*self.spriteToKm
            #rint(eagle.position[1],"Eagle km:",km,self.orbit_height)
            self.alt = self.orbit_height + km   #km is negative from orbit start
            #rint(self.alt)
            if self.alt < 0.02:
              #rint("Getting close")
              shadow.origin = list(self.origin_pos)
              #shadow.position[1] = 2*lp - eagle.position[1] # mirror from ground level
              self.serviceLander("eagle_fireball",eagle,lp)
              self.modeFrac = 1
              self.adjScalePos(1,2) 
            elif km > 0:
              self.modeFrac = math.pow(km/self.max_orbit_height,0.5)
              print("Mode frac:",self.modeFrac)
              self.adjScalePos(1,0)
              #endif    
            else:
              #self.modeFrac = math.pow(km/(0.02-self.orbit_height),0.5)
              self.modeFrac = km/(0.02-self.orbit_height)
              #rint("Mode frac:",self.modeFrac)
              self.adjScalePos(1,2)
              shadow.scale = 0.001*self.modeFrac
              shadow.origin = list(self.origin_pos)
              #shadow.position[1] = self.sprites["moon"].position[1]+self.modeFrac
              #shadow.origin[1] = 0
              #shadow.position[1] = 0   #/shadow.scale
            #endif
            shadow.position[1] = (self.sprites["moon"].position[1]+self.modeFrac)*(1-self.modeFrac) + (2*lp - eagle.position[1])*self.modeFrac
          #endif          
        #endif  
        true_alt = self.alt*1000  #-(land_pos*self.spriteToKm))*1000
        #print("land pos:",land_pos,land_pos*self.spriteToKm)
        feet_alt = true_alt*3.28084
        #feet_alt = 1100
        hundreds = feet_alt/100 % 10
        thousands = feet_alt/1000 % 10
        self.sprites["altimeter_minute"].angle = hundreds*36
        self.sprites["altimeter_hour"].angle = thousands*36
        #print("hundreds:",hundreds)

        if descent.divorced: #not connected to flys by itself
          if descent.down:
            self.serviceCrash("descent_fireball",descent,land_pos,frame_time)
          else:
            self.service(descent,frame_time)
            self.serviceLander("descent_fireball",descent,lp)
          #endif
          if command.position[0] < eagle.position[0] - 500:
            command.position[0] = eagle.position[0] - 500
          elif command.position[0] > eagle.position[0] + 500:
            command.position[0] = eagle.position[0] + 500
          #endif  
        #endif  
        self.service(vehicle,frame_time)
        orbit_speed_frac = -vehicle.vel[0]/self.orbit_speed  
        ground_speed = (1-orbit_speed_frac)  
        #rint("Ground speed:",ground_speed)
        self.moon.moon_angle += ground_speed*frame_time*0.1  
    #enddef
    
    def serviceLander(self,nm,target,lp):
      true_alt = target.position[1]-lp
      if true_alt < 0  and not target.down:  #touch down
        target.down = True
        print("Touch down:",target.vel, target.position[1],lp)
        target.position[1] = lp 
        dv = target.vel[0] + self.orbit_speed
        #rint("dv:",dv)
        target.rot_vel = -dv
        if dv > 10 or target.vel[1] < -10:
          fb = self.sprites[nm]
          fb.scale = 0.1
          fb.visible = True
          self.showCrashBlurb = True
          #rint("Show crash blurb!")
          target.vel[1] = 0
          shadow = self.sprites['shadow']
          shadow.visible = False
        else:
          self.showSuccessBlurb = True
          target.vel[1] = target.vel[1]*-0.2  #bounce with some damping
        #endif
    #enddef
    
    def serviceCrash(self,nm,target,land_pos,frame_time):
      target.position[1] = land_pos + self.origin_pos[1]
      dv = target.vel[0] + self.orbit_speed
      target.angle %= 360
      da = target.angle - 270
      if da > 180:
        da -= 360
      elif da < -180:
        da += 360
      #endif
      shadow = self.sprites['shadow']

      #rint("dv:",dv, target.vel[0],self.orbit_speed)
      if abs(dv) > 10 or abs(da) > 45:  # too fast or wrong way up
        target.rot_vel = -dv*0.3
        target.vel[0] -= dv*frame_time*0.1  # 10 seconds to stop
      else:
        target.vel[0] = -self.orbit_speed

        #rint("Target angle:",target.angle,da)  # zero is pointing right
        if da > 10:
          target.rot_vel -= frame_time*0.1  #right the eagle on its base
        elif da < -10:
          target.rot_vel += frame_time*0.1
        else:
          target.rot_vel = 0
          target.angle = 270
          if target.name == "eagle":
            shadow.visible = True
            shadow.position[1] = target.position[1]+0.1
          #endif   
        #endif
      #endif

      fb = self.sprites[nm]
      if fb.visible:
        fb.rot_vel = target.rot_vel
        #fb.position = eagle.position
        #rint("FB:",fb.position,fb.origin," Eagle:",eagle.position,eagle.origin)
        fb.angle = target.angle
        if abs(dv) > 10:
          #if fb.scale < 0.5:
          #  fb.scale += fb.scale*frame_time
          #endif
          fb.scale = 0.5*random.random()
        else:
          if abs(dv) > 1:
            fb.scale = 0.1*dv*random.random()
          elif fb.scale != 10:
            fb.scale = 10 # blast
          else:  
            shadow.visible = True
            shadow.position[1] = target.position[1]+0.1
            fb.visible = False
            target.visible = False
          #endif         
        #endif
      #endif    
    #enddef
        
    def service(self,vehicle,frame_time):
      angle = vehicle.angle
      cos = math.cos(math.radians(angle))
      sin = math.sin(math.radians(angle))
      ax = vehicle.acc[0]*cos + vehicle.acc[1]*sin
      ay = vehicle.acc[1]*cos - vehicle.acc[0]*sin
      #print(cos,sin,ax,ay)
      gx = self.moon_gravity*vehicle.vel[0]/self.orbit_speed
      #rint("Vehicle:",vehicle.name,gx,vehicle.vel[1],self.orbit_speed)
      vehicle.vel[0] += ax*frame_time
      if not vehicle.down:
        vehicle.vel[1] += (ay+gx)*frame_time
      #rint(vehicle.name,vehicle.acc,"vel:",vehicle.vel)
      ss = frame_time*self.kmToScreen
      vehicle.position[0] += vehicle.vel[0]*ss
      vehicle.position[1] += vehicle.vel[1]*ss
      #rint("Vehicle:",vehicle.name,vehicle.position[1], self.orbit_height*self.kmToScreen)
      '''if vehicle.position[1] < -self.orbit_height*self.kmToSprite:
        #rint("Before:",vehicle.position[1]," nm:",vehicle.name)
        vehicle.position[1] = -self.orbit_height*self.kmToSprite
        vehicle.vel[1] = vehicle.vel[1]*-0.2  #bounce with some damping
        #rint("After:",vehicle.position[1]," nm:",vehicle.name)
      #endif'''
      vehicle.angle -= math.degrees(vehicle.rot_vel*frame_time)
    #enddef  
       
    def createOffScreenSurface(self):
        # Create a 24bit (rgba) offscreen surface pygame can render to
        self.pg_screen = pygame.Surface(self.window_size, flags=pygame.SRCALPHA)
        # 24 bit (rgba) moderngl texture
        self.pg_texture = self.ctx.texture(self.window_size, 4)
        self.pg_texture.filter = moderngl.NEAREST, moderngl.NEAREST

        self.texture_program = self.load_program('texture.glsl')
        self.quad_fs = geometry.quad_fs()
        pygame.font.init()
        #rint(pygame.font.get_fonts())
        self.font = pygame.font.Font(None,24)
        #rint(dir(self.font))
        self.font2 = pygame.font.Font(None,48)
        self.load_gui_elements()
    #enddef    
        
    def print(self, txt, loc, col = (255,255,255), font = None):
      if font == None:
        font = self.font
      #endif
      text = font.render(txt, True, col)
      self.pg_screen.blit(text, loc) 
    #end 
        
    def render_pygame(self, time, fps):
        command = self.sprites["command"]
        eagle = self.sprites["eagle"]
        descent = self.sprites["descent"]
        """Render to offscreen surface and copy result into moderngl texture"""
        self.pg_screen.fill((0, 0, 0, 0))  # Make sure we clear with alpha 0!
        dy = self.font.get_linesize()*1.2
        yo = self.font.get_linesize()*0.2
        ym = self.font.get_linesize()*0.1
        if self.pause:
          rc = (0,0,0)
          self.sprite_win.height = (len(self.sprites)+1)*dy
          #self.sprite_win.bottom = self.window_size[1]-10
          #rint("SW:",self.sprite_win)
          pygame.draw.rect(self.pg_screen,(255,255,255),self.sprite_win)
          y = self.sprite_win.top
          num_select = 0
          sprite_selected = None
          
          for i,sprite in enumerate(self.sprites.values()):
            if sprite.selected:
              num_select += 1
              sprite_selected = sprite
              pygame.draw.rect(self.pg_screen,(100,255,255),(self.sprite_win.left,y,self.sprite_win.width,dy))
            #endif
            self.print(sprite.name,(self.sprite_win.left,y+yo),(0,0,0),self.font)
            if sprite.visible:
              img_nm = "eye_open"
            else:
              img_nm = "eye_closed"  
            #endif  
            self.pg_screen.blit(self.gui[img_nm],(self.sprite_win.left+150,y-ym))
            if sprite.saved:
              img_nm = "saved"
            else:
              img_nm = "unsaved"  
            #endif  
            self.pg_screen.blit(self.gui[img_nm],(self.sprite_win.left+120,y-ym))
            y += dy
          #endif
          if num_select == 1:
            status_win = pygame.Rect(250,10,self.window_size[0]-260,dy)
            pygame.draw.rect(self.pg_screen,(255,255,0),status_win)         
            self.print(sprite_selected.name + ":" + sprite_selected.getStatus(),(status_win.left,status_win.top),(0,0,0),self.font)
          #endif
        #endif
        if len(self.errors) > 0:
          error_win = pygame.Rect(250,50,
            self.window_size[0]-260,min(len(self.errors)*dy,self.window_size[1]-60))
          #rint("EW:",error_win)
          pygame.draw.rect(self.pg_screen,(164,164,164),error_win)
          y = error_win.top
          for err in self.errors:
            self.print(err,(error_win.left,y),(0,0,255),self.font)
            y += dy
          #endfor
        #endif
        if eagle.divorced: 
          eagle_vel = (self.orbit_speed + self.sprites["eagle"].vel[0])*3.6
          if self.alt < 1:
            alt_str = "Alt:%5.0fm"%(self.alt*1000,)
          else:  
            alt_str = "Alt:%5.2fkm"%(self.alt,)
          #endif  
          self.print("Speed %4.0fkm/hr "%(eagle_vel,)+alt_str,(20,20),(128,255,128),self.font2)
          if descent.divorced:
            fuel_frac = self.eagle_fuel/self.eagle_init_fuel
          else:
            fuel_frac = self.descent_fuel/self.descent_init_fuel
          #endif    
        else:  
          self.print("Moon %0.3f Alt:%0.1f"%(self.moon.moon_scale,self.moon.moon_pos[1]),(0,self.window_size[1]-dy),(255,255,255),self.font)
          fuel_frac = self.command_fuel/self.command_init_fuel
        #endif
        self.print("Fuel  Altitude",(self.window_size[0]-120,20),(255,255,255),self.font)
        alt_bar = (self.window_size[1]-100)*(1-self.modeFrac)
        fuel_bar = (self.window_size[1]-100)*fuel_frac
        pygame.draw.rect(self.pg_screen,(255,0,0),(self.window_size[0]-50,self.window_size[1]-alt_bar-50,20,alt_bar))
        pygame.draw.rect(self.pg_screen,(0,0,255),(self.window_size[0]-100,self.window_size[1]-fuel_bar-50,20,fuel_bar))
        pygame.draw.rect(self.pg_screen,(128,128,128),(self.window_size[0]-120,self.window_size[1]-50,120,20))
        if self.showCrashBlurb:
          self.print("Material from this crash was ejected into lunar orbit",(150,100),(0,0,255),self.font2)
          self.print("making it unsafe to return for the next million years",(150,150),(0,0,255),self.font2)
          if self.joystick_server:
            self.print("Press buttons 6 to 11 to reset",(400,250),(128,128,0),self.font2)
          else:
            self.print("Press End key to reset",(400,250),(128,128,0),self.font2)
          #endif  
        elif self.showSuccessBlurb:
          self.print("Congratuations! You have landed safely on the moon",(180,100),(0,255,0),self.font2)
          if self.joystick_server:
            self.print("Press button 3 to attempt redocking with command module (impossible!)",(150,150),(255,0,0),self.font2)
            self.print("Press buttons 6 to 11 to reset",(400,250),(128,128,0),self.font2)
          else:
            self.print("Press Home key to attempt redocking with command module  (impossible!)",(150,150),(255,0,0),self.font2)
            self.print("Press End key to reset",(400,250),(128,128,0),self.font2)
          #endif  
        elif self.showWait:
          self.print("Please wait while we get the moon turning.",(400,self.window_size[1]-150),(0,255,255),self.font2)        
        elif self.showInstructions:
          self.print("Attempt to land on the moon.",(400,self.window_size[1]-150),(0,255,255),self.font2)
          self.print("Orbit height and orbit speed have been reduced for expedience",(100,self.window_size[1]-100),(0,128,0),self.font2)
          if self.joystick_server:
            self.print("Press button 2 to undock from command module",(200,self.window_size[1]-50),(255,255,0),self.font2)
          else:
            self.print("Press numpad zero key to undock from command module",(200,self.window_size[1]-50),(255,255,0),self.font2) 
           
        #endif
        if not eagle.divorced and self.joysticks > 0:
          self.pg_screen.blit(self.joyimg,(0,0))  #self.window_size[0] - self.joyimg.size[0],self.window_size[1] - self.joyimg.size[1])
        pg_screen = pygame.transform.flip(self.pg_screen, False, True)  # Flip the text vertically.  
        texture_data = pg_screen.get_view('1')
        self.pg_texture.write(texture_data)
    #enddef
    

    def key_event(self, key, action, modifiers): 
        super().key_event(key, action, modifiers)
        print("Key:",key," Mods:",modifiers)
        left_shift_key = 65505
        left_cntrl_key = 65507
        left_alt_key   = 65513
        right_shift_key = 65506
        right_cntrl_key = 65508
        right_alt_key   = 65514

        keys = self.wnd.keys
        print("SLASH:",keys.SLASH)
        #rint(dir(keys))
        #'A', 'ACTION_PRESS', 'ACTION_RELEASE', 'B', 'BACKSLASH', 'BACKSPACE', 'C', 'CAPS_LOCK', 'COMMA', 'D', 'DELETE', 'DOWN', 'E', 'END', 'ENTER', 'EQUAL', 'ESCAPE', 'F', 'F1', 'F10', 'F11', 'F12', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G', 'H', 'HOME', 'I', 'INSERT', 'J', 'K', 'L', 'LEFT', 'LEFT_BRACKET', 'M', 'MINUS', 'N', 'NUMBER_0', 'NUMBER_1', 'NUMBER_2', 'NUMBER_3', 'NUMBER_4', 'NUMBER_5', 'NUMBER_6', 'NUMBER_7', 'NUMBER_8', 'NUMBER_9', 'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9', 'O', 'P', 'PAGE_DOWN', 'PAGE_UP', 'PERIOD', 'Q', 'R', 'RIGHT', 'RIGHT_BRACKET', 'S', 'SEMICOLON', 'SLASH', 'SPACE', 'T', 'TAB', 'U', 'up', 'V', 'W', 'X', 'Y', 'Z'
        mods = self.wnd.modifiers
        #rint(dir(mods))
        #'alt', 'ctrl', 'shift'
        dpos = 100
        dangle = 45
        if mods.ctrl:
          dpos /= 10
          dangle /= 2
        #endif
        if mods.shift:
          dpos /= 10
          dangle /= 2
        #endif
        #rint("dpos:",dpos)
        if action == keys.ACTION_PRESS:
          if key == left_shift_key or key == right_shift_key:
            self.shift_pressed = True
          elif key == left_cntrl_key or key == right_cntrl_key:
            self.cntrl_pressed = True
          elif key == left_alt_key or key == right_alt_key:
            self.alt_pressed = True
          #endif
          #rint("Mods:",self.shift_pressed,self.cntrl_pressed,self.alt_pressed)
          if key == keys.SPACE:
            self.pause = not self.pause
          #endif  
          if self.pause:
            for i, sprite in enumerate(self.sprites.values()):
              #rint(i,sprite)
              mask = 1 << i
              if sprite.selected:
                if self.arrow_key_mode == 0:
                  if key == keys.LEFT:
                    sprite.angle -= dangle
                  elif key == keys.RIGHT:
                    sprite.angle += dangle
                  elif key == keys.UP:
                    sprite.scale *= 0.9
                  elif key == keys.DOWN:
                    sprite.scale *= 1.1
                  #endif
                else:   
                  if key == keys.LEFT:
                    sprite.position[0] -= dpos
                  elif key == keys.RIGHT:
                    sprite.position[0] += dpos
                  elif key == keys.UP:
                    sprite.position[1] += dpos
                  elif key == keys.DOWN:
                    sprite.position[1] -= dpos
                  #endif
                #endif
                if key == keys.D:
                  if sprite.divorced:
                    sprite.undivorce()
                  else:  
                    sprite.divorce(self.aspect)
                  #endif  
                  #rint("D:",sprite.divorced)
                #endif  
              #endif
            #endfor
            if key == keys.I:
              self.moon.radius.value *= 0.999
              print("radius:",self.moon.radius.value)
            elif key == keys.O:
              self.moon.radius.value *= 1.001
              print("radius:",self.moon.radius.value)
            elif key == keys.COMMA:
              self.modeChange = 1
              self.modeFrac = 0
            elif key == keys.PERIOD:
              self.modeChange = 2
              self.modeFrac = 0
            elif key == keys.SLASH:
              self.modeChange = 3
              self.modeFrac = 0
            elif key == keys.S:
              if mods.ctrl:
                self.saveConfig()
              #endif  
            elif key == keys.H:
              self.showhidden = not self.showhidden
              for sprite in self.sprites.values():
                if sprite.selected:
                  sprite.showBackground(self.showhidden)
                #endif
              #endfor
            elif key == keys.L:
              self.showlocs = not self.showlocs
              for sprite in self.sprites.values():
                if sprite.selected:
                  sprite.showLocs(self.showlocs)
                #endif
              #endfor
            elif key == keys.M:
              self.arrow_key_mode = 1 - self.arrow_key_mode
            elif key == keys.HOME:
              self.moon.adj_pos(0,0.1)
            elif key == keys.END:
              self.moon.adj_pos(0,-0.1)
            elif key == keys.PAGE_UP:
              self.moon.adj_scale(1.1)
            elif key == keys.PAGE_DOWN:
              self.moon.adj_scale(0.9)
            elif key == keys.NUMBER_1:
              self.sprites[0].selected = not self.sprites[0].selected 
            elif key == keys.NUMBER_2:
              self.sprites[1].selected = not self.sprites[1].selected 
            elif key == keys.A:
              if mods.ctrl:
                for sp in self.sprites.values():
                  sp.selected = True
                  sp.showBackground(True)
                  sp.showLocs(True)
                #endfor
              elif mods.shift:
                for sp in self.sprites.values():
                  sp.selected = False
                  sp.showBackground(False)
                  sp.showLocs(False)
                #endfor
              #endif
            #endif  
          else:        
            if key == keys.NUMPAD_1:
              self.rocket_left_dn = 1
            elif key == keys.NUMPAD_3:
              self.rocket_right_dn = 1
            elif key == keys.NUMPAD_4:
              self.rocket_left = 1
            elif key == keys.NUMPAD_6:
              self.rocket_right = 1
            elif key == keys.NUMPAD_7:
              self.rocket_left_up = 1
            elif key == keys.NUMPAD_9:
              self.rocket_right_up = 1
            elif key == keys.NUMPAD_8:
              self.rocket_descent = 1
            elif key == keys.NUMPAD_5:
              rocket_eagle = 1
            elif key == keys.NUMPAD_2:
              self.rocket_command = 1
            elif key == keys.NUMPAD_0:
              self.sprites['eagle'].divorce(self.aspect)
              self.sprites["shadow"].visible = True
            elif key == keys.HOME:
              self.sprites['descent'].divorce(self.aspect)
              self.sprites['eagle'].down = False
              self.showSuccess = False
            elif key == keys.END:
              self.reset()
            #endif
          #endif
        elif action == self.wnd.keys.ACTION_RELEASE:
          if key == left_shift_key or key == right_shift_key:
            self.shift_pressed = False
          elif key == left_cntrl_key or key == right_cntrl_key:
            self.cntrl_pressed = False
          elif key == left_alt_key or key == right_alt_key:
            self.alt_pressed = False
          elif key == keys.NUMPAD_1:
            self.rocket_left_dn = 0
          elif key == keys.NUMPAD_3:
            self.rocket_right_dn = 0
          elif key == keys.NUMPAD_4:
            self.rocket_left = 0
          elif key == keys.NUMPAD_6:
            self.rocket_right = 0
          elif key == keys.NUMPAD_7:
            self.rocket_left_up = 0
          elif key == keys.NUMPAD_9:
            self.rocket_right_up = 0
          elif key == keys.NUMPAD_8:
            self.rocket_descent = 0
          elif key == keys.NUMPAD_5:
            self.rocket_eagle = 0
          elif key == keys.NUMPAD_2:
            self.rocket_command= 0
          #endif
        #endif
        print("Mods:",self.shift_pressed,self.cntrl_pressed,self.alt_pressed)
    #enddef
    
    def getObjectPos(self, x, y, set_loc = False):
        sx = 2*x/self.window_size[0] - 1
        sy = 2*y/self.window_size[1] - 1
        print("Screen pos:",sx,sy)
        near_sprite = None
        near_loc = None
        near_dist = 1000
        for sprite in self.sprites.values():
          print("sp:"+ sprite.name,sprite.selected,sprite.divorced)
          if not sprite.selected:
            continue
          #endif
          if sprite.parent and not sprite.divorced:  #can't get object position if it's not divorced from parent ******feature to add oneday
            sprite.logError("Can't find pos on child:"+sprite.name+" Divorce first")
            continue
          #endif
          p = sprite.screenToSprite((sx,sy))
          if set_loc:
            sprite.setLoc(p)
            sprite.saved = False
            sprite.showLocs(True)
          else:
            dist, loc = sprite.nearest(p,near_dist)
            if loc:
              near_loc = loc
              near_sprite = sprite
            #endif  
          #endif    
          print("Sprite:",sprite.name,p)
        #endfor
        return near_sprite, near_loc
    #enddef        
    
    def mouse_position_event(self, x, y, dx, dy):
        #print("Mouse position:", x, y, dx, dy)
        pass
    #enddef

    def mouse_drag_event(self, x, y, dx, dy):
        #print("Mouse drag:", x, y, dx, dy)
        pass
    #enddef

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        print("Mouse wheel:", x_offset, y_offset)
    #enddef
    
    def load_gui_elements(self):
        imgs = ("eye_open","eye_closed","saved","unsaved")
        for nm in imgs:
          img = pygame.image.load("images/"+nm+".png")
          self.gui[nm] = img #pygame.transform.flip(img, False, True)
        #endfor  
    #enddef 
    
    def select_none(self):
      for sp in self.sprites.values():
        sp.selected = False
        sp.showBackground(False)
        sp.showLocs(False)
      #endfor  
    #enddef

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))       
        if self.sprite_win.collidepoint(x,y):
          wx = x - self.sprite_win.left
          wy = int((y - self.sprite_win.top)/self.font.get_linesize()/1.2)
          print(wx,wy)
          if wy >= len(self.sprites):
            self.select_none()
            return
          sprite = list(self.sprites.values())[wy]
          if wx < 130:
            if self.cntrl_pressed:
              sprite.selected = not sprite.selected
            else:
              self.select_none()
              sprite.selected = True
              sprite.showBackground(True)
              sprite.showLocs(True)
            #endif
          elif wx < 150:
            sprite.save()
          elif wx < 174:
            print("Toggle ",wy," visibility")
            sprite.visible = not sprite.visible    
          return
        #endif  
        if button == 1:
          near_sprite,near_loc = self.getObjectPos(x,y)
          if near_sprite != None:
            print("Press Near:",near_sprite.name,near_loc)
          #endif          
        elif button == 2:
          self.getObjectPos(x,y,set_loc = True)
        #endif  
    #enddef

    def mouse_release_event(self, x: int, y: int, button: int):
        print("Mouse button {} released at {}, {}".format(button, x, y))
        if self.sprite_win.collidepoint(x,y):
          return
        #endif  
        if button == 1:
          near_sprite,near_loc = self.getObjectPos(x,y)
          if near_sprite != None:
            print("Release Near:",near_sprite.name,near_loc)
          #endif          
    #enddef

    def adjScalePos(self, i0, i1, pos_adj = 0):
        ms = 1000*self.moon.radius.value  #34740
        moon_scales = (1.0,4.939,100)   #,ms,89.55)
        moon_position = (0.0,-3.943,0.5-100+pos_adj)  #,-ms,-88.64)   #***************work on moon position
        #sprite_scales = (0.001,0.001,0.001)
        moon_img_scales = (0.00024,0.001195,0.024)  #,0.24,0.0217)
        moon_img_position = (0.0,-3.943,0.5-100+pos_adj)  #,-ms,-88.654)
        
        #sprite_position = (0,0.6,0.6)
        sprite = self.sprites["command"]
        moon = self.sprites["moon"]
        self.moon.moon_scale = interp(self.modeFrac,moon_scales[i0],moon_scales[i1])
        self.moon.moon_pos[1] = interp(self.modeFrac,moon_position[i0],moon_position[i1])  # / self.moon.moon_scale
        self.moon.set_pos()
        self.moon.adj_scale(1.0)
        #rint("Moon    pos:",self.moon.moon_pos,self.moon.moon_scale)
        #sprite.scale = interp(self.modeFrac,sprite_scales[i0],sprite_scales[i1])
        #sprite.position[1] = interp(self.modeFrac,sprite_position[i0],sprite_position[i1]) / sprite.scale
        moon.scale = interp(self.modeFrac,moon_img_scales[i0],moon_img_scales[i1])
        moon.position[1] = interp(self.modeFrac,moon_img_position[i0],moon_img_position[i1]) / moon.scale
        #rint("Moon img pos:",moon.position[1]*moon.scale,moon.scale)
    #enddef
      
#endclass
    
if __name__ == '__main__':
    game.run()
