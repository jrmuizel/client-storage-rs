extern crate gleam;
extern crate glutin;
extern crate libc;
extern crate euclid;
extern crate core_foundation;

use euclid::{Transform3D, Vector3D};

use gleam::gl::Gl;
use gleam::gl;
use glutin::GlContext;

use std::rc::Rc;

use gleam::gl::GLuint;
use gleam::gl::ErrorCheckingGl;
use core_foundation::dictionary::CFMutableDictionary;

struct Options {
    pbo: bool,
    client_storage: bool,
    texture_array: bool,
    texture_storage: bool,
    swizzle: bool,
}


fn init_shader_program(gl: &Rc<Gl>, vs_source: &[u8], fs_source: &[u8]) -> gl::GLuint {
    let vertex_shader = load_shader(gl, gl::VERTEX_SHADER, vs_source);
    let fragment_shader = load_shader(gl, gl::FRAGMENT_SHADER, fs_source);
    let shader_program = gl.create_program();
    gl.attach_shader(shader_program, vertex_shader);
    gl.attach_shader(shader_program, fragment_shader);
    gl.link_program(shader_program);

    let mut link_status = [0];
    unsafe {
        gl.get_program_iv(shader_program, gl::LINK_STATUS, &mut link_status);
            if link_status[0] == 0 {
                println!("LINK: {}", gl.get_program_info_log(shader_program));
            }

    }
    shader_program
}

struct Buffers {
    position: GLuint,
    texture_coord: GLuint,
    indices: GLuint
}

fn init_buffers(gl: &Rc<gl::Gl>, texture_rectangle: bool, texture_width: i32, texture_height: i32) -> Buffers {
    let position_buffer = gl.gen_buffers(1)[0];

    gl.bind_buffer(gl::ARRAY_BUFFER, position_buffer);

    let positions = [
        // Front face
        -1.0f32, -1.0,  1.0,
        1.0, -1.0,  1.0,
        1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,

        // Back face
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0, -1.0, -1.0,

        // Top face
        -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        1.0,  1.0, -1.0,

        // Bottom face
        -1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,

        // Right face
        1.0, -1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0,  1.0,  1.0,
        1.0, -1.0,  1.0,

        // Left face
        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0,  1.0,
        -1.0,  1.0, -1.0,
    ];


    gl.buffer_data_untyped(gl::ARRAY_BUFFER, std::mem::size_of_val(&positions) as isize, positions.as_ptr() as *const libc::c_void, gl::STATIC_DRAW);


    let texture_coord_buffer = gl.gen_buffers(1)[0];

    gl.bind_buffer(gl::ARRAY_BUFFER, texture_coord_buffer);

    let width = if texture_rectangle { texture_width as f32 } else { 1.0 };
    let height = if texture_rectangle { texture_height as f32 } else { 1.0 };

    let texture_coordinates = [
        // Front
        0.0f32,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Back
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Top
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Bottom
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Right
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
        // Left
        0.0,  0.0,
        width,  0.0,
        width,  height,
        0.0,  height,
    ];

    gl.buffer_data_untyped(gl::ARRAY_BUFFER, std::mem::size_of_val(&texture_coordinates) as isize, texture_coordinates.as_ptr() as *const libc::c_void, gl::STATIC_DRAW);

    // Build the element array buffer; this specifies the indices
    // into the vertex arrays for each face's vertices.

    let index_buffer = gl.gen_buffers(1)[0];

    gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);

    // This array defines each face as two triangles, using the
    // indices into the vertex array to specify each triangle's
    // position.

    let indices = [
        0u16,  1,  2,      0,  2,  3,    // front
        4,  5,  6,      4,  6,  7,    // back
        8,  9,  10,     8,  10, 11,   // top
        12, 13, 14,     12, 14, 15,   // bottom
        16, 17, 18,     16, 18, 19,   // right
        20, 21, 22,     20, 22, 23,   // left
    ];

    // Now send the element array to GL

    gl.buffer_data_untyped(gl::ELEMENT_ARRAY_BUFFER, std::mem::size_of_val(&indices) as isize, indices.as_ptr() as *const libc::c_void, gl::STATIC_DRAW);


    Buffers {
        position: position_buffer,
        texture_coord: texture_coord_buffer,
        indices: index_buffer,
    }
}

struct Image {
    data: Vec<u8>,
    width: i32,
    height: i32
}

fn rgba_to_bgra(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);
    let mut i = 0;
    while i < buf.len() {
        let r = buf[i];
        let g = buf[i+1];
        let b = buf[i+2];
        let a = buf[i+3];
        buf[i] = b;
        buf[i+1] = g;
        buf[i+2] = r;
        buf[i+3] = a;
        i += 4;
    }
}

fn make_blue(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);
    let mut i = 0;
    while i < buf.len() {
        buf[i] = 0xff;
        buf[i+1] = 0;
        buf[i+2] = 0;
        buf[i+3] = 0xff;
        i += 4;
    }
}

fn make_yellow(buf: &mut [u8]) {
    assert!(buf.len() % 4 == 0);
    let mut i = 0;
    while i < buf.len() {
        buf[i] = 0;
        buf[i+1] = 0xff;
        buf[i+2] = 0xff;
        buf[i+3] = 0xff;
        i += 4;
    }
}

fn paint_square(image: &mut Image) {
    let width = image.width as usize;
    for i in 1024..2048 {
        make_yellow(&mut image.data[i*width..(i*width + 512)]);
    }

}

fn paint_square2(image: &mut Image) {
    let width = image.width as usize;
    for i in 1024..2048 {
        make_blue(&mut image.data[i*width..(i*width + 512)]);
    }

}

fn bpp(format: GLuint) -> i32 {
    match format {
        gl::UNSIGNED_INT_8_8_8_8_REV => 4,
        gl::UNSIGNED_BYTE => 4,
        gl::FLOAT => 16,
        gl::INT => 16,
        gl::UNSIGNED_INT => 16,
        _ => panic!()
    }
}



fn load_image(format: GLuint) -> Image
{
    if true {
        let width: i32 = 4096;
        let height: i32 = 8192;
        return Image { data: vec![0; (width * height * bpp(format)) as usize], width, height }
    }
    let decoder = png::Decoder::new(std::fs::File::open("cubetexture.png").unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    // Allocate the output buffer.
    let mut buf = vec![0; info.buffer_size()];
    // Read the next frame. Currently this function should only called once.
    // The default options
    reader.next_frame(&mut buf).unwrap();

    rgba_to_bgra(&mut buf);

    //make_red(&mut buf);
    //make_yellow(&mut buf);

    Image { data: buf, width: info.width as i32, height: info.height as i32}
}

struct Texture {
    id: GLuint,
    pbo: Option<GLuint>,
}


fn load_texture(gl: &Rc<gl::Gl>, image: &Image, target: GLuint, internal_format: GLuint, src_format: GLuint, src_type: GLuint, options: &Options) -> Texture {
    let texture = gl.gen_textures(1)[0];

    gl.bind_texture(target, texture);

    let level = 0;
    let border = 0;

    if options.client_storage {
        //gl.texture_range_apple(target, &image.data[..]);

        // both of these seem to work ok on Intel
        let storage = gl::STORAGE_SHARED_APPLE;
        let storage = gl::STORAGE_CACHED_APPLE;
        gl.tex_parameter_i(target, gl::TEXTURE_STORAGE_HINT_APPLE, storage as gl::GLint);
        gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);

        // this may not be needed
        gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0);
    }

    if options.texture_array {
        if options.texture_storage {
            gl.tex_storage_3d(target, 1, internal_format, image.width as i32, image.height as i32, 1);
        } else {
            gl.tex_image_3d(target, level, internal_format as i32, image.width, image.height, 1, border, src_format, src_type, if options.pbo { None } else { Some(&image.data[..]) });
        }
    } else {
        if options.texture_storage {
            gl.tex_storage_2d(target, 1, internal_format, image.width as i32, image.height as i32);
        } else {
            gl.tex_image_2d(target, level, internal_format as i32, image.width, image.height, border, src_format, src_type, if options.pbo { None } else { Some(&image.data[..]) });
        }
    }

    // Rectangle textures has its limitations compared to using POT textures, for example,
    // Rectangle textures can't use mipmap filtering
    gl.tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);

    // Rectangle textures can't use the GL_REPEAT warp mode
    gl.tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
    gl.tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);
    if options.swizzle {
        //let components = [gl::RED, gl::GREEN, gl::BLUE, gl::ALPHA];
        let components = [gl::BLUE, gl::GREEN, gl::RED, gl::ALPHA];
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_R, components[0] as i32);
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_G, components[1] as i32);
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_B, components[2] as i32);
        gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_A, components[3] as i32);
    }

    let pbo = if options.pbo {
        let id = gl.gen_buffers(1)[0];
        // WebRender on Mac uses DYNAMIC_DRAW
        gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, id);
        gl.buffer_data_untyped(gl::PIXEL_UNPACK_BUFFER, (image.width * image.height * bpp(src_type)) as _, image.data[..].as_ptr() as *const libc::c_void, gl::DYNAMIC_DRAW);

        if options.texture_array {
            gl.tex_sub_image_3d_pbo(
                target,
                level,
                0,
                0,
                0,
                image.width,
                image.height,
                1,
                src_format,
                src_type,
                0,
            );
        } else {
            gl.tex_sub_image_2d_pbo(
                target,
                level,
                0,
                0,
                image.width,
                image.height,
                src_format,
                src_type,
                0,
            );
        }
        gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
        Some(id)
    } else {
        None
    };
    Texture { id: texture, pbo }
}


fn load_shader(gl: &Rc<Gl>, shader_type: gl::GLenum, source: &[u8]) -> gl::GLuint {
    let shader = gl.create_shader(shader_type);
    gl.shader_source(shader, &[source]);
    gl.compile_shader(shader);
    let mut status = [0];
    unsafe { gl.get_shader_iv(shader, gl::COMPILE_STATUS, &mut status); }
    if status[0] == 0 {
        println!("{}", gl.get_shader_info_log(shader));
        panic!();
    }
    return shader;
}

fn allow_gpu_switching() {
    use core_foundation::string::CFString;
    use core_foundation::base::TCFType;
    use core_foundation::boolean::CFBoolean;

    let b = core_foundation::bundle::CFBundle::main_bundle();
    let mut i : CFMutableDictionary<_, _> = unsafe { std::mem::transmute(b.info_dictionary()) };
    i.set(CFString::new("NSSupportsAutomaticGraphicsSwitching"), CFBoolean::true_value().into_CFType());
}

fn main() {

    allow_gpu_switching();

    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768);
    let context = glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_gl(glutin::GlRequest::GlThenGles {
        opengl_version: (3, 2),
        opengles_version: (3, 0),
    });

    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    let options = Options { pbo: false, client_storage: true, texture_array: false, texture_storage: false, swizzle: false };

    let texture_rectangle = true;
    let apple_format = true; // on Intel it looks like we don't need this particular format

    let texture_target = if texture_rectangle { gl::TEXTURE_RECTANGLE_ARB } else { gl::TEXTURE_2D };
    let texture_target = if options.texture_array { gl::TEXTURE_2D_ARRAY} else { texture_target };

    //let texture_internal_format = gl::RGBA32UI;
    //let texture_internal_format = gl::RGBA32F;
    let texture_internal_format = gl::RGBA8;

    let mut texture_src_format = if apple_format { gl::BGRA } else { gl::RGBA };
    let mut texture_src_type = if apple_format { gl::UNSIGNED_INT_8_8_8_8_REV } else { gl::UNSIGNED_BYTE };

    // adjust type and format to match internal format
    if texture_internal_format == gl::RGBA32UI {
        texture_src_type = gl::UNSIGNED_INT;
        texture_src_format = gl::RGBA_INTEGER;
    } else if texture_internal_format == gl::RGBA32F {
        texture_src_format = gl::RGBA;
        texture_src_type = gl::FLOAT;
    }


    let vs_source = b"
    #version 140

    in vec4 a_vertex_position;
    in vec2 a_texture_coord;
    uniform mat4 u_model_view_matrix;
    uniform mat4 u_projection_matrix;
    out vec2 v_texture_coord;
    void main(void) {
        gl_Position = u_projection_matrix * u_model_view_matrix * a_vertex_position;
        v_texture_coord = a_texture_coord;
    }";

    let (sampler, coord) = if texture_rectangle {
        ("sampler2DRect", "v_texture_coord")
    } else if options.texture_array {
        ("sampler2DArray", "vec3(v_texture_coord, 0.0)")
    } else {
        ("sampler2D", "v_texture_coord")
    };
    let fs_source = format!("
    #version 140

    in vec2 v_texture_coord;
    uniform {} u_sampler;
    out vec4 fragment_color;
    void main(void) {{
        fragment_color = texture(u_sampler, {});
    }}
    ", sampler, coord).into_bytes();


    let mut glc = unsafe { gl::GlFns::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _) };
    let gl = ErrorCheckingGl::wrap(glc);// Rc::get_mut(&mut glc).unwrap();

    let shader_program = init_shader_program(&gl, vs_source, &fs_source);


    let vertex_position = gl.get_attrib_location(shader_program, "a_vertex_position");
    let texture_coord = gl.get_attrib_location(shader_program, "a_texture_coord");

    let projection_matrix_loc = gl.get_uniform_location(shader_program, "u_projection_matrix");
    let model_view_matrix_loc = gl.get_uniform_location(shader_program, "u_model_view_matrix");
    let u_sampler = gl.get_uniform_location(shader_program, "u_sampler");


    let mut image = load_image(texture_src_type);
    let buffers = init_buffers(&gl, texture_rectangle, image.width, image.height);


    let texture = load_texture(&gl, &image,
                               texture_target,
                               texture_internal_format,
                               texture_src_format,
                               texture_src_type,
                               &options);

    let vao = gl.gen_vertex_arrays(1)[0];
    gl.bind_vertex_array(vao);


    let mut running = true;
    let mut cube_rotation: f32 = 0.;
    while running {
        events_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent{ event, .. } => match event {
                    glutin::WindowEvent::Closed => running = false,
                    glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                    _ => ()
                },
                _ => ()
            }
        });


        // Bind the texture to texture unit 0
        gl.bind_texture(texture_target, texture.id);

        {
            let level = 0;
            if options.pbo {
                let id = texture.pbo.unwrap();
                gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, id);
                if true {
                    //gl.buffer_sub_data_untyped(gl::PIXEL_UNPACK_BUFFER, 0, (image.width * image.height * 4) as _, image.data[..].as_ptr() as *const libc::c_void);
                    gl.buffer_data_untyped(gl::PIXEL_UNPACK_BUFFER, (image.width * image.height * bpp(texture_src_type)) as _, image.data[..].as_ptr() as *const libc::c_void, gl::DYNAMIC_DRAW);

                } else {
                    //gl.map_buffer(gl::PIXEL_UNPACK_BUFFER, gl::WRITE_ONLY);
                }
                if options.texture_array {
                    gl.tex_sub_image_3d_pbo(
                        texture_target,
                        level,
                        0,
                        0,
                        0,
                        image.width,
                        image.height,
                        1,
                        texture_src_format,
                        texture_src_type,
                        0,
                    );
                } else {
                    gl.tex_sub_image_2d_pbo(
                        texture_target,
                        level,
                        0,
                        0,
                        image.width,
                        image.height,
                        texture_src_format,
                        texture_src_type,
                        0,
                    );
                }
                gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
            } else {
                if options.client_storage {
                    //gl.tex_parameter_i(texture_target, gl::TEXTURE_STORAGE_HINT_APPLE, gl::STORAGE_CACHED_APPLE as gl::GLint);
                    //gl.pixel_store_i(gl::UNPACK_CLIENT_STORAGE_APPLE, true as gl::GLint);
                }
                gl.tex_sub_image_2d(texture_target, level, 0, 0, image.width, image.height, texture_src_format, texture_src_type, &image.data[..]);
                // sub image uploads are still fast as long as the memory is in the same place
                //gl.tex_sub_image_2d(texture_target, level, 0, 256, image.width, 4096, texture_src_format, texture_src_type, &image.data[image.width as usize *4*256..image.width as usize *4*4096]);

            }
        }


        gl.clear_color(1., 0., 0., 1.);
        gl.clear_depth(1.);
        gl.enable(gl::DEPTH_TEST);
        gl.depth_func(gl::LEQUAL);

        gl.clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

        // Create a perspective matrix, a special matrix that is
        // used to simulate the distortion of perspective in a camera.
        // Our field of view is 45 degrees, with a width/height
        // ratio that matches the display size of the canvas
        // and we only want to see objects between 0.1 units
        // and 100 units away from the camera.

        let field_of_view = 45. * std::f32::consts::PI / 180.;   // in radians
        let width = 1024.;
        let height = 768.;
        let aspect = width / height;
        let z_near = 0.1;
        let z_far = 100.0;

        let fovy = field_of_view;
        let near = z_near;
        let far = z_far;
        let f = 1. / (fovy / 2.).tan();
        let nf = 1. / (near - far);

        let projection_matrix = Transform3D::<f32>::row_major(
            f / aspect, 0., 0., 0.,
            0.,  f, 0., 0.,
            0., 0.,  (far + near) * nf, -1.,
            0., 0., 2. * far * near * nf, 0.
        );

        let mut model_view_matrix = Transform3D::<f32>::identity();
        model_view_matrix = model_view_matrix.post_translate(Vector3D::new(-0., 0., -6.0));
        model_view_matrix = model_view_matrix.pre_rotate(0., 0., 1., euclid::Angle::radians(cube_rotation));
        model_view_matrix = model_view_matrix.pre_rotate(0., 1., 0., euclid::Angle::radians(cube_rotation * 0.7));

        {
            let num_components = 3;
            let ty = gl::FLOAT;
            let normalize = false;
            let stride = 0;
            let offset = 0;
            gl.bind_buffer(gl::ARRAY_BUFFER, buffers.position);
            gl.vertex_attrib_pointer(vertex_position as u32, num_components, ty, normalize, stride, offset);
            gl.enable_vertex_attrib_array(vertex_position as u32);
        }

        {
            let num_components = 2;
            let ty = gl::FLOAT;
            let normalize = false;
            let stride = 0;
            let offset = 0;
            gl.bind_buffer(gl::ARRAY_BUFFER, buffers.texture_coord);
            gl.vertex_attrib_pointer(texture_coord as u32, num_components, ty, normalize, stride, offset);
            gl.enable_vertex_attrib_array(texture_coord as u32);
        }

        gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, buffers.indices);

        gl.use_program(shader_program);

        gl.uniform_matrix_4fv(
            projection_matrix_loc,
            false,
            &projection_matrix.to_row_major_array());

        gl.uniform_matrix_4fv(
            model_view_matrix_loc,
            false,
            &model_view_matrix.to_row_major_array());

        // Specify the texture to map onto the faces.

        // Tell OpenGL we want to affect texture unit 0
        gl.active_texture(gl::TEXTURE0);


        gl.uniform_1i(u_sampler, 0);

        {
            let vertex_count = 36;
            let ty = gl::UNSIGNED_SHORT;
            let offset = 0;
            gl.draw_elements(gl::TRIANGLES, vertex_count, ty, offset);
        }
        //paint_square(&mut image);
        gl_window.swap_buffers().unwrap();
        /*
        let mut count = 0;
        gl.finish_object_apple(gl::TEXTURE, texture.id);
        while gl.test_object_apple(gl::TEXTURE, texture.id) == 0 {
            count += 1
        }
                println!("test {}", count);

        */
        //paint_square2(&mut image);




        cube_rotation += 0.1;
    }
}
