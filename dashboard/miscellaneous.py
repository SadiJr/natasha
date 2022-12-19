from collections import namedtuple

from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State


HtmlElement = namedtuple("HtmlElement", ["id", "property"])


class Dropdown:
    style = {"border": 'grey solid', 'padding': '5px', 'margin': '5px'}

    def __init__(self, header, dropdown_id, dropdown_objects, include_refresh_button):
        self.header = header
        self.dropdown_id = dropdown_id
        self.dropdown_objects = dropdown_objects
        self.include_refresh_button = include_refresh_button

    def generate_dash_element(self):
        return html.Div([
            html.H5(self.header, id=f'{self.dropdown_id}_heading'),
            dcc.Dropdown(
                id=self.dropdown_id,
                options=[{'label': name, 'value': name} for name, _ in self.dropdown_objects.items()]
            ),
            html.P("Opções:", style={'padding': '5px', 'margin': '5px'}),
            html.Div(id=f'{self.dropdown_id}_options', style=self.style),
            html.Div(html.Button("Atualizar", id=f'{self.dropdown_id}_refresh', style=self.style))
        ], id=f'{self.dropdown_id}_div')

    @property
    def dropdown_args(self):
        return '%s' % self.dropdown_id, 'value'

    @property
    def refresh_args(self):
        return '%s_refresh' % self.dropdown_id, 'n_clicks'

    @property
    def options_args(self):
        return '%s_options' % self.dropdown_id, 'children'

    def get_input(self, element='dropdown'):
        return Input(*getattr(self, f"{element}_args"))

    def get_state(self, element='dropdown'):
        return State(*getattr(self, f"{element}_args"))

    def generate_update_options_callback(self, app):
        @app.callback(
            Output(*self.options_args),
            [self.get_input('dropdown')]
        )
        def update_options(dropdown_choice):
            if dropdown_choice is None:
                return
            return self.opt_elements(dropdown_choice)

    def opt_elements(self, dropdown_choice):
        if dropdown_choice is None:
            return

        return [
            *[e
              for option_name, default_value in self.dropdown_objects[dropdown_choice].get_options().items()
              for e in (f"{option_name}: ", dcc.Input(id="%s|%s" % (dropdown_choice, option_name), type="text",
                                                      value=str(default_value)))],
        ]

    def options(self, options_element):
        opt = {}

        for e in options_element:
            if not isinstance(e, dict):
                continue

            id_pros, value = e["props"]["id"], tuple(e["props"]["value"].strip("()").split(","))
            if len(value) == 1:
                value = value[0]
            opt[id_pros.split("|")[1]] = value

        return opt

    def apply(self, dropdown_choice, options_element, df):
        return self.dropdown_objects[dropdown_choice](**self.options(options_element)).apply(df)


def datatable(df, table_id, max_rows=10, max_cell_width="600px", text_overflow="ellipsis"):
    if df is None:
        return dash_table.DataTable(id=table_id)

    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": str(i), "id": str(i)} for i in df.columns],
        data=df[:max_rows].to_dict("records"),
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
        style_table={
            'border': 'thin lightgrey solid', 'maxHeight': '350px', 'overflowY': 'auto',
        },
        style_cell={
            'minWidth': '0px', 'whiteSpace': 'no-wrap', 'overflow': 'hidden',
            'textOverflow': text_overflow, 'maxWidth': max_cell_width,
        }
    )


def c_picker(df, ele_id):
    if df is None:
        return dcc.Dropdown(id=ele_id)

    return dcc.Dropdown(
        id=ele_id, value=[], multi=True, options=[{'label': col, 'value': col} for col in df.columns]
    )

