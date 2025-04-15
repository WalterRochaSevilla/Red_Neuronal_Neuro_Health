-- Tabla Usuario
CREATE TABLE Usuario (
    id_usuario INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    contraseña VARCHAR(255) NOT NULL,
    fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP,
    rol ENUM('usuario', 'especialista', 'administrador') NOT NULL
);

-- Tabla Chatbot
CREATE TABLE Chatbot (
    id_interaccion INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    mensaje TEXT NOT NULL,
    respuesta TEXT NOT NULL,
    fecha_interaccion DATETIME DEFAULT CURRENT_TIMESTAMP,
    emocion_detectada VARCHAR(50),
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario)
);

-- Tabla Especialista
CREATE TABLE Especialista (
    id_especialista INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    especialidad VARCHAR(100) NOT NULL,
    descripcion TEXT,
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario)
);

-- Tabla Citas
CREATE TABLE Citas (
    id_cita INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    id_especialista INT,
    fecha_cita DATETIME NOT NULL,
    estado ENUM('pendiente', 'completada', 'cancelada') DEFAULT 'pendiente',
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario),
    FOREIGN KEY (id_especialista) REFERENCES Especialista(id_especialista)
);

-- Tabla Exámenes
CREATE TABLE Exámenes (
    id_examen INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    tipo_examen VARCHAR(100) NOT NULL,
    resultado TEXT NOT NULL,
    fecha_examen DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario)
);

-- Tabla Notas Emocionales
CREATE TABLE Notas_Emocionales (
    id_nota INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    emocion VARCHAR(50) NOT NULL,
    fecha_nota DATETIME DEFAULT CURRENT_TIMESTAMP,
    comentario TEXT,
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario)
);

-- Tabla Foro
CREATE TABLE Foro (
    id_post INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    titulo VARCHAR(200) NOT NULL,
    contenido TEXT NOT NULL,
    fecha_publicacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario)
);

-- Tabla Meditaciones
CREATE TABLE Meditaciones (
    id_meditacion INT AUTO_INCREMENT PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    descripcion TEXT,
    audio_url VARCHAR(255) NOT NULL,
    duracion INT NOT NULL
);

-- Tabla Historial (Vista del Especialista)
CREATE TABLE Historial (
    id_historial INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT,
    id_especialista INT,
    fecha_consulta DATETIME DEFAULT CURRENT_TIMESTAMP,
    notas TEXT,
    FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario),
    FOREIGN KEY (id_especialista) REFERENCES Especialista(id_especialista)
);