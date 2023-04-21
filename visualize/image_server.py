from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from argparse import ArgumentParser
from http import HTTPStatus
import datetime
import email.utils
import html
import io
import mimetypes
import posixpath
import shutil
import sys
import urllib.parse
from pathlib import Path

__version__ = "0.0.1"

IMAGE_EXT = ['.jpg', '.jpeg', '.png', ',gif', '.bmp', '.webp']


class SimpleImageHTTPServer(BaseHTTPRequestHandler):
    server_version = "SimpleImageHTTPServer/" + __version__
    protocol_version = "HTTP/1.0"

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = Path(directory or global_args.path)

        if not mimetypes.inited:
            mimetypes.init()
        self.extensions_map = mimetypes.types_map.copy()
        self.extensions_map.update({'': 'application/octet-stream'})
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_url.query)

        main_path = self.translate_path(parsed_url.path)
        cmp_path = query.get("c", None)

        if not main_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Directory not exists")
            return

        f = None
        if main_path.is_file():
            f = self.mode_file(main_path)
        elif main_path.is_dir() or main_path.is_symlink():
            if cmp_path:
                cmp_path = [main_path] + [self.translate_path(x) for x in cmp_path]
                f = self.mode_compare(cmp_path)
            else:
                f = self.mode_directory(main_path)

        if f:
            try:
                shutil.copyfileobj(f, self.wfile)
            finally:
                f.close()

    def get_image_file_list(self, path: Path):
        return sorted([x for x in path.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXT],
                      key=lambda f: f.name.lower())

    def mode_compare(self, paths: list):
        r = list()
        paths_ok = list()
        for d in paths:
            if not d.exists() or d.is_file():
                r.append("<p>Ignore path \"{}\".</p>".format(
                    html.escape(str(d.relative_to(self.directory)), quote=False)))
            else:
                paths_ok.append(d)
        images = [self.get_image_file_list(x) for x in paths_ok]
        images = [{f.name: f for f in d} for d in images]
        images_names = set()
        for x in images:
            images_names.update(x.keys())
        images_names = sorted(list(images_names))
        r.append("<table>")
        r.append("<thead>")
        r.append("<tr>")
        r.append("<td>Name</td>")
        for filename in paths_ok:
            r.append("<td>Directory: {}</td>".format(
                html.escape(str(filename.relative_to(self.directory)), quote=False)))
        r.append("</tr>")
        r.append("</thead>")
        r.append("<tbody>")
        for filename in images_names:
            r.append("<tr>")
            r.append("<td>{}</td>".format(filename))
            for d in images:
                if filename in d.keys():
                    r.append('<td><a href="/{fnm}" target="_blank"><img src="/{fnm}" /></a></td>'.format(
                        fnm=html.escape(str(d[filename].relative_to(self.directory)), quote=False)))
                else:
                    r.append("<td>None</td>")
            r.append("<tr>")
        r.append("</tbody>")
        r.append("</table>")
        return self.make_html("\n".join(r))

    def mode_directory(self, src: Path):
        r = list()
        subdir = sorted([x for x in src.iterdir() if x.is_dir()], key=lambda d: d.name.lower())
        images = self.get_image_file_list(src)
        if subdir:
            r.append('<ul>')
            for d in subdir:
                r.append('<li><a href="{0}/">{1}/</a></li>'.format(
                    urllib.parse.quote(d.name, errors='surrogatepass'),
                    html.escape(d.name, quote=False)))
            r.append('</ul>')
        for file in images:
            r.append('<a href="/{0}" target="_blank"><img src="/{0}" /></a>'.format(
                html.escape(str(file.relative_to(self.directory)), quote=False)))

        return self.make_html("\n".join(r))

    def mode_file(self, path: Path):
        try:
            f = path.open('rb')
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            ctype = self.guess_type(path)
            fs = path.stat()
            # Use browser cache if possible
            if ("If-Modified-Since" in self.headers and "If-None-Match" not in self.headers):
                # compare If-Modified-Since and time of last file modification
                try:
                    ims = email.utils.parsedate_to_datetime(self.headers["If-Modified-Since"])
                except (TypeError, IndexError, OverflowError, ValueError):
                    # ignore ill-formed values
                    pass
                else:
                    if ims.tzinfo is None:
                        # obsolete format with no timezone, cf.
                        # https://tools.ietf.org/html/rfc7231#section-7.1.1.1
                        ims = ims.replace(tzinfo=datetime.timezone.utc)
                    if ims.tzinfo is datetime.timezone.utc:
                        # compare to UTC datetime of last modification
                        last_modif = datetime.datetime.fromtimestamp(fs.st_mtime, datetime.timezone.utc)
                        # remove microseconds, like in If-Modified-Since
                        last_modif = last_modif.replace(microsecond=0)

                        if last_modif <= ims:
                            self.send_response(HTTPStatus.NOT_MODIFIED)
                            self.end_headers()
                            f.close()
                            return None

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(fs.st_size))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except:
            f.close()
            raise

    def make_html(self, content: str):
        enc = sys.getfilesystemencoding()
        css = ["<style>", "</style>"]
        img_style = "img{{ {border}{size} }}".format(
            border="border:1px dashed blue;" if global_args.border else "",
            size="max-height:{size}px;max-width:{size}px".format(size=global_args.size) if global_args.size else ""
        )
        table_style = "table, th, td { border: 1px solid #eee; border-collapse: collapse; text-align:center; }" \
                      "thead tr th:first-child, tbody tr td:first-child {" \
                      "    width: 8em;min-width: 8em;max-width: 8em; word-break: break-all;}" \
                      "table{ width: 100%; } " \
                      "table tr:nth-child(even) { background-color: #fcfcfc; }" \
                      "table tr:nth-child(odd) { background-color: #fff; }"
        css.insert(-1, img_style)
        css.insert(-1, table_style)
        r = """
            <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
            <html>
            <head>
                <meta http-equiv="Content-Type" content="text/html; charset={charset}">
                <title>ImageViewer</title>
                {css}
            </head>
            <body>

            {content}

            </body>
            </html>
            """.format(charset=enc, css="\n".join(css), content=content or "Nothing here")

        encoded = r.encode(enc, 'surrogateescape')
        f = io.BytesIO()
        f.write(encoded)
        f.seek(0)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return f

    def translate_path(self, path: str):
        try:
            path = urllib.parse.unquote(path, errors='surrogatepass')
        except UnicodeDecodeError:
            path = urllib.parse.unquote(path)
        path = path.lstrip("/")
        path = self.directory / Path(path)
        return path

    def guess_type(self, path):
        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']


if __name__ == '__main__':
    parser = ArgumentParser(description="Simple Image HTTP Server")
    parser.add_argument('--port', type=int, default=8007)
    parser.add_argument('--path', type=str, default="./")
    parser.add_argument('--border', action="store_true")
    parser.add_argument('--size', type=int, default=500)
    global_args = parser.parse_args()

    print("{:-^50}".format(" Let's rock "))

    with ThreadingHTTPServer(("", global_args.port), SimpleImageHTTPServer) as httpd:
        sa = httpd.socket.getsockname()
        print("Serving on {host} port {port} (http://{host}:{port}/) ...".format(host=sa[0], port=sa[1]))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")

    print("{:-^50}".format(" All is well "))
